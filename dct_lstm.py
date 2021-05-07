# LSTM layers with DCT-parameterized weights

import torch
import math

import numpy as np

import torch_dct as dct
import torch.nn.functional as F
import torch.nn as nn

from external_torch_dct import DCTLayer


# LSTM layer with DCT-parameterized weights
class DctLSTM(nn.Module):
    '''LSTM with weights genereted by DCT related ops.'''

    def __init__(self, input_dim, hidden_dim, sparsity_ih, sparsity_hh,
                 weight_drop=0.0, dropout_dct=False, cuda=True):
        super(DctLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_ih = sparsity_ih
        self.sparsity_hh = sparsity_hh
        self.weight_drop = weight_drop
        self.dropout_dct = dropout_dct
        self.cuda = cuda

        if weight_drop > 0.0:
            self.wdrop = nn.Dropout(weight_drop)

        in_coeffs_dim, in_num_diags = self.get_sparse_config(
            input_dim, hidden_dim, sparsity_ih)

        hidden_coeffs_dim, hidden_num_diags = self.get_sparse_config(
            hidden_dim, hidden_dim, sparsity_hh)

        self.in_dct_layer = DCTLayer(
            in_features=input_dim, type='idct', norm='ortho', cuda=cuda)
        self.hid_dct_layer = DCTLayer(
            in_features=hidden_dim, type='idct', norm='ortho', cuda=cuda)

        # number of diagonals
        self.in_num_diags = in_num_diags
        self.hidden_num_diags = hidden_num_diags

        # number of coefficients
        self.in_coeffs_dim = in_coeffs_dim
        self.hidden_coeffs_dim = hidden_coeffs_dim

        self.coeffs_iig = nn.Parameter(
            self.get_dct_init(
                in_coeffs_dim, hidden_dim, input_dim, in_num_diags))
        self.coeffs_ifg = nn.Parameter(
            self.get_dct_init(
                in_coeffs_dim, hidden_dim, input_dim, in_num_diags))
        self.coeffs_ihg = nn.Parameter(
            self.get_dct_init(
                in_coeffs_dim, hidden_dim, input_dim, in_num_diags))
        self.coeffs_iog = nn.Parameter(
            self.get_dct_init(
                in_coeffs_dim, hidden_dim, input_dim, in_num_diags))

        self.coeffs_hig = nn.Parameter(
            self.get_dct_init(
                hidden_coeffs_dim, hidden_dim, hidden_dim, hidden_num_diags))
        self.coeffs_hfg = nn.Parameter(
            self.get_dct_init(
                hidden_coeffs_dim, hidden_dim, hidden_dim, hidden_num_diags))
        self.coeffs_hhg = nn.Parameter(
            self.get_dct_init(
                hidden_coeffs_dim, hidden_dim, hidden_dim, hidden_num_diags))
        self.coeffs_hog = nn.Parameter(
            self.get_dct_init(
                hidden_coeffs_dim, hidden_dim, hidden_dim, hidden_num_diags))

        self.in_coeffss = [
            self.coeffs_iig, self.coeffs_ifg, self.coeffs_ihg, self.coeffs_iog]
        self.hid_coeffss = [
            self.coeffs_hig, self.coeffs_hfg, self.coeffs_hhg, self.coeffs_hog]

        bias_init = torch.rand([hidden_dim * 4])
        initrange = 1.0 / math.sqrt(hidden_dim)
        nn.init.uniform_(bias_init, -initrange, initrange)
        self.bias = nn.Parameter(bias_init)

    def get_dct_init(self, len_coeffs, dim_out, dim_in, diag_shift):

        factor = 1.

        init = torch.rand([dim_out, dim_in])
        if self.cuda:  # TODO update to device.
            init = init.cuda()

        initrange = 1.0 / math.sqrt(dim_out)
        # initrange = 0.1
        nn.init.uniform_(init, -initrange, initrange)
        init_f = torch.fliplr(dct.dct_2d(init, norm='ortho'))
        ind = torch.triu_indices(dim_out, dim_in, diag_shift)
        coeffs_init = init_f[tuple(ind)] * factor

        return coeffs_init

    def to_weights(self, coeffs, ind, zero_weights, linear1, linear2):

        weights = torch.fliplr(zero_weights.index_put_(tuple(ind), coeffs))
        weights = linear1(weights)
        weights = linear2(weights.transpose(-1, -2))
        return weights.transpose(-1, -2)

    def get_sparse_config(self, in_dim, out_dim, sparsity_level):
        '''Get num_diagonals and num coeffs.

        Given the dimension of matrix
          in_dim: number of columns
          out_dim: number of rows

        We want to find the right diagonal shift "d" s.t.

        N(d) < thr(desired sparsity) < N(d+1)
        N(d+1)

        We search as follows:
        - If: N(0) is below thr: try N(n) for n = -1..-out_dim
        - Else: try N(n) for n = 1..in_dim

        input: 2 dimensions of the weight matrix
        output: tuple (num_diagonal, num_coeff)
        '''

        total_el = in_dim * out_dim
        thr = int(total_el * (1 - sparsity_level))  # just truncate fraction.

        for num_diag in range(in_dim):  # upper triagular matrix.
            non_zeros = torch.triu_indices(out_dim, in_dim, num_diag).size()[1]
            if non_zeros < thr:
                break

        if num_diag == 0:  # also check the other direction
            for neg_diag in range(-1, -out_dim, -1):
                new_non_zeros = torch.triu_indices(
                    out_dim, in_dim, neg_diag).size()[1]
                if new_non_zeros > thr:
                    # means that the previous one was the good one.
                    break
                else:
                    non_zeros = new_non_zeros
                    num_diag = neg_diag

        print(f"sparsity: {(total_el - non_zeros) / total_el * 100 :.1f} %"
              f" vs. desired sparsity {sparsity_level * 100} %")
        return non_zeros, num_diag

    def get_weights(self, device):
        # Generate the full weights.
        # return: weights of shape (hidden_dim * 4 , input_dim * hidden_dim)

        # input to hidden
        w_ih = None
        ind = torch.triu_indices(
            self.hidden_dim, self.input_dim, self.in_num_diags, device=device)
        weights_f = torch.zeros(
            [self.hidden_dim, self.input_dim], device=device)
        for coeffs in self.in_coeffss:
            if self.dropout_dct:
                coeffs = self.wdrop(coeffs)

            weights = self.to_weights(
                coeffs, ind, weights_f, self.in_dct_layer, self.hid_dct_layer)

            if w_ih is not None:
                w_ih = torch.cat([w_ih, weights], dim=0)
            else:
                w_ih = weights

        # hidden to hidden
        w_hh = None
        ind = torch.triu_indices(
            self.hidden_dim, self.hidden_dim, self.hidden_num_diags,
            device=device)
        weights_f = torch.zeros(
            [self.hidden_dim, self.hidden_dim], device=device)
        for coeffs in self.hid_coeffss:
            if self.dropout_dct:
                coeffs = self.wdrop(coeffs)

            weights = self.to_weights(
                coeffs, ind, weights_f, self.hid_dct_layer, self.hid_dct_layer)

            if w_hh is not None:
                w_hh = torch.cat([w_hh, weights], dim=0)
            else:
                w_hh = weights

        # concatenate both
        weights = torch.cat([w_ih, w_hh], dim=1)
        return weights

    def get_dense_weights(self):
        # Generate the full weights.
        # return: weights of shape (hidden_dim * 4 , input_dim * hidden_dim)

        # input to hidden
        w_ih = None
        for coeffs in self.in_coeffss:
            weights = coeffs.view(self.hidden_dim, self.input_dim)

            if w_ih is not None:
                w_ih = torch.cat([w_ih, weights], dim=0)
            else:
                w_ih = weights

        # hidden to hidden
        w_hh = None
        for coeffs in self.hid_coeffss:
            weights = coeffs.view(self.hidden_dim, self.hidden_dim)

            if w_hh is not None:
                w_hh = torch.cat([w_hh, weights], dim=0)
            else:
                w_hh = weights

        # concatenate both
        weights = torch.cat([w_ih, w_hh], dim=1)
        return weights

    def forward(self, input_, hidden=None, device='cuda'):
        # input shape: (len, B, dim)
        # output shape: (len * B, num_classes)
        outputs = []

        weights = self.get_weights(device)
        if self.weight_drop > 0.0:
            weights = self.wdrop(weights)

        for x in torch.unbind(input_, dim=0):
            h = self.forward_step(x, hidden, weights)
            outputs.append(h[0].clone())
            hidden = h[1]
        op = torch.squeeze(torch.stack(outputs))
        return op, hidden

    def forward_step(self, x, prev_state, weights=None, device='cpu'):
        assert weights is not None
        # One time step forwarding.
        # input x: (B,  in_dim)
        # prev_state: tuple 2 * (B, out_dim)

        if prev_state is None:
            h = torch.zeros([x.size()[0], self.hidden_dim], device=device)
            c = torch.zeros([x.size()[0], self.hidden_dim], device=device)
        else:
            h, c = torch.squeeze(prev_state[0]), torch.squeeze(prev_state[1])

        out = torch.cat([x, h], dim=1)
        out = F.linear(out, weights, self.bias)
        h_out, i_out, f_out, o_out = torch.split(
            out, [self.hidden_dim, self.hidden_dim, self.hidden_dim,
                  self.hidden_dim], dim=1)

        h_out = torch.tanh(h_out)
        i_out = torch.sigmoid(i_out)
        f_out = torch.sigmoid(f_out)
        o_out = torch.sigmoid(o_out)

        if c is None:
            c = i_out * h_out
        else:
            c = i_out * h_out + f_out * c
        out = o_out * torch.tanh(c)
        return out, (out, c)


# LSTM layer with DCT-parameterized weights;
# low frequency coefficients are non zero.
class LowFreqDctLSTM(nn.Module):
    '''LSTMCell with weights genereted by DCT related ops.'''

    def __init__(self, input_dim, hidden_dim, sparsity_ih, sparsity_hh,
                 weight_drop=0.0, dropout_dct=False, cuda=True):
        super(LowFreqDctLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_ih = sparsity_ih
        self.sparsity_hh = sparsity_hh
        self.weight_drop = weight_drop
        self.dropout_dct = dropout_dct
        self.cuda = cuda

        if weight_drop > 0.0:
            self.wdrop = nn.Dropout(weight_drop)

        in_coeffs_dim, in_num_diags = self.get_sparse_config(
            input_dim, hidden_dim, sparsity_ih)

        hidden_coeffs_dim, hidden_num_diags = self.get_sparse_config(
            hidden_dim, hidden_dim, sparsity_hh)

        self.in_dct_layer = DCTLayer(
            in_features=input_dim, type='idct', norm='ortho', cuda=cuda)
        self.hid_dct_layer = DCTLayer(
            in_features=hidden_dim, type='idct', norm='ortho', cuda=cuda)

        # number of diagonals
        self.in_num_diags = in_num_diags
        self.hidden_num_diags = hidden_num_diags

        # number of coefficients
        self.in_coeffs_dim = in_coeffs_dim
        self.hidden_coeffs_dim = hidden_coeffs_dim

        self.coeffs_iig = nn.Parameter(
            self.get_dct_init(
                in_coeffs_dim, hidden_dim, input_dim, in_num_diags))
        self.coeffs_ifg = nn.Parameter(
            self.get_dct_init(
                in_coeffs_dim, hidden_dim, input_dim, in_num_diags))
        self.coeffs_ihg = nn.Parameter(
            self.get_dct_init(
                in_coeffs_dim, hidden_dim, input_dim, in_num_diags))
        self.coeffs_iog = nn.Parameter(
            self.get_dct_init(
                in_coeffs_dim, hidden_dim, input_dim, in_num_diags))

        self.coeffs_hig = nn.Parameter(
            self.get_dct_init(
                hidden_coeffs_dim, hidden_dim, hidden_dim, hidden_num_diags))
        self.coeffs_hfg = nn.Parameter(
            self.get_dct_init(
                hidden_coeffs_dim, hidden_dim, hidden_dim, hidden_num_diags))
        self.coeffs_hhg = nn.Parameter(
            self.get_dct_init(
                hidden_coeffs_dim, hidden_dim, hidden_dim, hidden_num_diags))
        self.coeffs_hog = nn.Parameter(
            self.get_dct_init(
                hidden_coeffs_dim, hidden_dim, hidden_dim, hidden_num_diags))

        self.in_coeffss = [
            self.coeffs_iig, self.coeffs_ifg, self.coeffs_ihg, self.coeffs_iog]
        self.hid_coeffss = [
            self.coeffs_hig, self.coeffs_hfg, self.coeffs_hhg, self.coeffs_hog]

        bias_init = torch.rand([hidden_dim * 4])
        initrange = 1.0 / math.sqrt(hidden_dim)
        nn.init.uniform_(bias_init, -initrange, initrange)
        self.bias = nn.Parameter(bias_init)

    def get_dct_init(self, len_coeffs, dim_out, dim_in, diag_shift):

        factor = 1.
        init = torch.rand([dim_out, dim_in])
        if self.cuda:  # old
            init = init.cuda()

        initrange = 1.0 / math.sqrt(dim_out)
        nn.init.uniform_(init, -initrange, initrange)
        init_f = torch.fliplr(dct.dct_2d(init, norm='ortho'))
        ind = torch.tril_indices(dim_out, dim_in, -diag_shift)
        coeffs_init = init_f[tuple(ind)] * factor

        return coeffs_init

    def to_weights(self, coeffs, ind, zero_weights, linear1, linear2):

        weights = torch.fliplr(zero_weights.index_put_(tuple(ind), coeffs))
        weights = linear1(weights)
        weights = linear2(weights.transpose(-1, -2))
        return weights.transpose(-1, -2)

    def get_sparse_config(self, in_dim, out_dim, sparsity_level):
        '''Get num_diagonals and num coeffs.

        Given the dimension of matrix
          in_dim: number of columns
          out_dim: number of rows

        We want to find the right diagonal shift "d" s.t.

        N(d) < thr(desired sparsity) < N(d+1)
        N(d+1)

        We search as follows:
        - If: N(0) is below thr: try N(n) for n = -1..-out_dim
        - Else: try N(n) for n = 1..in_dim

        input: 2 dimensions of the weight matrix
        output: tuple (num_diagonal, num_coeff)
        '''

        total_el = in_dim * out_dim
        thr = int(total_el * (1 - sparsity_level))  # just truncate fraction.

        for num_diag in range(out_dim):  # upper triagular matrix.
            non_zeros = torch.tril_indices(
                out_dim, in_dim, -num_diag).size()[1]
            if non_zeros < thr:
                break

        print(f"sparsity: {(total_el - non_zeros) / total_el * 100 :.1f} %"
              f" vs. desired sparsity {sparsity_level * 100} %")
        return non_zeros, num_diag

    def get_weights(self, device):
        # Generate the full weights.
        # return: weights of shape (hidden_dim * 4 , input_dim * hidden_dim)

        # input to hidden
        w_ih = None
        ind = torch.tril_indices(
            self.hidden_dim, self.input_dim, -self.in_num_diags, device=device)
        weights_f = torch.zeros(
            [self.hidden_dim, self.input_dim], device=device)
        for coeffs in self.in_coeffss:
            if self.dropout_dct:
                coeffs = self.wdrop(coeffs)
            weights = self.to_weights(
                coeffs, ind, weights_f, self.in_dct_layer, self.hid_dct_layer)

            if w_ih is not None:
                w_ih = torch.cat([w_ih, weights], dim=0)
            else:
                w_ih = weights

        # hidden to hidden
        w_hh = None
        ind = torch.tril_indices(
            self.hidden_dim, self.hidden_dim, -self.hidden_num_diags,
            device=device)
        weights_f = torch.zeros(
            [self.hidden_dim, self.hidden_dim], device=device)

        for coeffs in self.hid_coeffss:
            if self.dropout_dct:
                coeffs = self.wdrop(coeffs)

            weights = self.to_weights(
                coeffs, ind, weights_f, self.hid_dct_layer, self.hid_dct_layer)

            if w_hh is not None:
                w_hh = torch.cat([w_hh, weights], dim=0)
            else:
                w_hh = weights

        # concatenate both
        weights = torch.cat([w_ih, w_hh], dim=1)
        return weights

    def get_dense_weights(self):
        # Generate the full weights.
        # return: weights of shape (hidden_dim * 4 , input_dim * hidden_dim)

        # input to hidden
        w_ih = None
        for coeffs in self.in_coeffss:
            weights = coeffs.view(self.hidden_dim, self.input_dim)

            if w_ih is not None:
                w_ih = torch.cat([w_ih, weights], dim=0)
            else:
                w_ih = weights

        # hidden to hidden
        w_hh = None
        for coeffs in self.hid_coeffss:
            weights = coeffs.view(self.hidden_dim, self.hidden_dim)

            if w_hh is not None:
                w_hh = torch.cat([w_hh, weights], dim=0)
            else:
                w_hh = weights

        # concatenate both
        weights = torch.cat([w_ih, w_hh], dim=1)
        return weights

    def forward(self, input_, hidden=None, device='cuda'):
        # input shape: (len, B, dim)
        # output shape: (len * B, num_classes)
        outputs = []

        weights = self.get_weights(device)
        if self.weight_drop > 0.0:
            weights = self.wdrop(weights)

        for x in torch.unbind(input_, dim=0):
            h = self.forward_step(x, hidden, weights)
            outputs.append(h[0].clone())
            hidden = h[1]
        op = torch.squeeze(torch.stack(outputs))
        return op, hidden

    def forward_step(self, x, prev_state, weights=None, device='cpu'):
        assert weights is not None
        # One time step forwarding.
        # input x: (B,  in_dim)
        # prev_state: tuple 2 * (B, out_dim)

        if prev_state is None:
            h = torch.zeros([x.size()[0], self.hidden_dim]).to(device)
            c = torch.zeros([x.size()[0], self.hidden_dim]).to(device)
        else:
            h, c = torch.squeeze(prev_state[0]), torch.squeeze(prev_state[1])

        out = torch.cat([x, h], dim=1)
        out = F.linear(out, weights, self.bias)
        h_out, i_out, f_out, o_out = torch.split(
            out,
            [self.hidden_dim, self.hidden_dim, self.hidden_dim,
             self.hidden_dim],
            dim=1)

        h_out = torch.tanh(h_out)
        i_out = torch.sigmoid(i_out)
        f_out = torch.sigmoid(f_out)
        o_out = torch.sigmoid(o_out)

        if c is None:
            c = i_out * h_out
        else:
            c = i_out * h_out + f_out * c
        out = o_out * torch.tanh(c)
        return out, (out, c)


# LSTM layer with effective weights generated by applying DCT twice.
class DoubleDctLSTM(nn.Module):
    '''LSTMCell with weights genereted by DCT related ops.'''

    def __init__(self, input_dim, hidden_dim, sparsity_ih, sparsity_hh,
                 weight_drop=0.0, dropout_dct=False, cuda=True):
        super(DoubleDctLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_ih = sparsity_ih
        self.sparsity_hh = sparsity_hh
        self.weight_drop = weight_drop
        self.dropout_dct = dropout_dct
        self.cuda = cuda

        if weight_drop > 0.0:
            self.wdrop = nn.Dropout(weight_drop)

        (ih_small_in, ih_small_out, ih_medium_in, ih_medium_out,
         _) = self.get_double_dct_accumulate_config(
             input_dim, hidden_dim, sparsity_ih)

        (hh_small_in, hh_small_out, hh_medium_in, hh_medium_out,
         _) = self.get_double_dct_accumulate_config(
             hidden_dim, hidden_dim, sparsity_hh)
        # print(ih_small_in, ih_small_out, ih_medium_in, ih_medium_out)
        assert hh_small_out == hh_small_in == ih_small_out
        assert hh_medium_in == hh_medium_out == ih_medium_out

        self.ih_small_in = ih_small_in
        self.ih_small_out = ih_small_out
        self.ih_medium_in = ih_medium_in
        self.ih_medium_out = ih_medium_out

        self.hh_small_in = hh_small_in
        self.hh_small_out = hh_small_out
        self.hh_medium_in = hh_medium_in
        self.hh_medium_out = hh_medium_out

        # precompute indices:
        device = "cuda" if cuda else "cpu"
        # for ih
        out_gen, in_gen = ih_small_out, ih_small_in
        ind_med = [[i, j] for i in range(out_gen) for j in range(in_gen)]
        self.ih_ind_med = torch.tensor(ind_med).to(device).t()

        ind_large = [[i, j] for i in range(
            self.ih_medium_out) for j in range(self.ih_medium_in)]
        self.ih_ind_large = torch.tensor(ind_large).to(device).t()

        # for hh
        out_gen, in_gen = hh_small_out, hh_small_in
        ind_med = [[i, j] for i in range(out_gen) for j in range(in_gen)]
        self.hh_ind_med = torch.tensor(ind_med).to(device).t()
        ind_large = [[i, j] for i in range(
            self.hh_medium_out) for j in range(self.hh_medium_in)]
        self.hh_ind_large = torch.tensor(ind_large).to(device).t()

        # precompute zero weights
        self.ih_weights_f_med = torch.zeros(
            [self.ih_medium_out, self.ih_medium_in], device=device)
        self.ih_weights_f_large = torch.zeros(
            [self.hidden_dim, self.input_dim], device=device)

        self.hh_weights_f_med = torch.zeros(
            [self.hh_medium_out, self.hh_medium_in], device=device)
        self.hh_weights_f_large = torch.zeros(
            [self.hidden_dim, self.hidden_dim], device=device)

        # DCT layers
        self.in_dct_layer = DCTLayer(
            in_features=input_dim, type='idct', norm='ortho', cuda=cuda)
        self.hid_dct_layer = DCTLayer(
            in_features=hidden_dim, type='idct', norm='ortho', cuda=cuda)

        self.in_dct_layer_medium = DCTLayer(
            in_features=ih_medium_in, type='idct', norm='ortho', cuda=cuda)
        self.hid_dct_layer_medium = DCTLayer(
            in_features=ih_medium_out, type='idct', norm='ortho', cuda=cuda)

        # number of coefficients
        self.in_coeffs_dim = ih_small_in * ih_small_out
        self.hidden_coeffs_dim = hh_small_in * hh_small_out

        # Be careful! in the double DCT model, coeffss are 2D.
        self.coeffs_iig = nn.Parameter(
            self.get_dct_init(hidden_dim, input_dim, ih_medium_out,
                              ih_medium_in, ih_small_out, ih_small_in))
        self.coeffs_ifg = nn.Parameter(
            self.get_dct_init(hidden_dim, input_dim, ih_medium_out,
                              ih_medium_in, ih_small_out, ih_small_in))
        self.coeffs_ihg = nn.Parameter(
            self.get_dct_init(hidden_dim, input_dim, ih_medium_out,
                              ih_medium_in, ih_small_out, ih_small_in))
        self.coeffs_iog = nn.Parameter(
            self.get_dct_init(hidden_dim, input_dim, ih_medium_out,
                              ih_medium_in, ih_small_out, ih_small_in))

        self.coeffs_hig = nn.Parameter(
            self.get_dct_init(hidden_dim, hidden_dim, hh_medium_out,
                              hh_medium_in, hh_small_out, hh_small_in))
        self.coeffs_hfg = nn.Parameter(
            self.get_dct_init(hidden_dim, hidden_dim, hh_medium_out,
                              hh_medium_in, hh_small_out, hh_small_in))
        self.coeffs_hhg = nn.Parameter(
            self.get_dct_init(hidden_dim, hidden_dim, hh_medium_out,
                              hh_medium_in, hh_small_out, hh_small_in))
        self.coeffs_hog = nn.Parameter(
            self.get_dct_init(hidden_dim, hidden_dim, hh_medium_out,
                              hh_medium_in, hh_small_out, hh_small_in))

        self.in_coeffss = [self.coeffs_iig, self.coeffs_ifg, self.coeffs_ihg,
                           self.coeffs_iog]
        self.hid_coeffss = [self.coeffs_hig, self.coeffs_hfg, self.coeffs_hhg,
                            self.coeffs_hog]

        bias_init = torch.rand([hidden_dim * 4])
        initrange = 1.0 / math.sqrt(hidden_dim)
        nn.init.uniform_(bias_init, -initrange, initrange)
        self.bias = nn.Parameter(bias_init)

    def get_double_dct_accumulate_config(self, in_dim, out_dim, sparsity):
        # Apply sparsity at each level

        # level one
        reduction_factor = 1 - sparsity
        medium_in = round(in_dim * math.sqrt(reduction_factor))
        medium_out = round(out_dim * math.sqrt(reduction_factor))
        current_sparsity = 1 - medium_in * medium_out / (in_dim * out_dim)
        print(f"[{self}] Level 1 sparsity: {current_sparsity}, "
              f"#params: {medium_in * medium_out}")

        # level two
        small_in = round(medium_in * math.sqrt(reduction_factor))
        small_out = round(medium_out * math.sqrt(reduction_factor))

        final_sparsity = small_in * small_out / (in_dim * out_dim)
        print(f"[{self}] Final sparsity: {1 - final_sparsity}, "
              f"#params: {small_in * small_out}")

        return small_in, small_out, medium_in, medium_out, final_sparsity

    def get_dct_init(self, dim_out, dim_in, med_out, med_in,
                     small_out, small_in):

        factor = 1.
        # initialize as if dense matrix
        init = torch.rand([dim_out, dim_in])
        if self.cuda:  # TODO update to device.
            init = init.cuda()
        initrange = 1.0 / math.sqrt(dim_out)
        # initrange = 0.1
        nn.init.uniform_(init, -initrange, initrange)

        # apply DCT
        init_f = dct.dct_2d(init, norm='ortho')
        ind = np.array(
            [[i, j] for i in range(med_out) for j in range(med_in)]
        ).transpose()
        coeffs_init = init_f[ind] * factor
        coeffs_init = torch.reshape(coeffs_init, [med_out, med_in])

        # apply further DCT
        coeffs_init = dct.dct_2d(coeffs_init, norm='ortho')
        ind = np.array(
            [[i, j] for i in range(small_out) for j in range(small_in)]
        ).transpose()
        coeffs_init = coeffs_init[ind] * factor
        coeffs_init = torch.reshape(coeffs_init, [small_out, small_in])

        return coeffs_init

    def to_weights_rectangular(self, coeffs, indices, zero_weights,
                               linear1, linear2):
        # Assume coeffs of shape (gen_in, gen_out)
        weights = zero_weights.clone()
        weights = weights.index_put_(tuple(indices), coeffs.flatten()).float()

        weights = linear1(weights)
        weights = linear2(weights.t())

        return weights.t()

    def get_weights(self, device):
        # Generate the full weights.
        # return: weights of shape (hidden_dim * 4 , input_dim * hidden_dim)

        # input to hidden
        w_ih = None

        for coeffs in self.in_coeffss:
            if self.dropout_dct:
                coeffs = self.wdrop(coeffs)
            # round 1
            weights1 = self.to_weights_rectangular(
                coeffs, self.ih_ind_med, self.ih_weights_f_med,
                self.in_dct_layer_medium, self.hid_dct_layer_medium)
            # round 2
            weights = self.to_weights_rectangular(
                weights1, self.ih_ind_large, self.ih_weights_f_large,
                self.in_dct_layer, self.hid_dct_layer)

            if w_ih is not None:
                w_ih = torch.cat([w_ih, weights], dim=0)
            else:
                w_ih = weights

        # hidden to hidden
        w_hh = None

        for coeffs in self.hid_coeffss:
            if self.dropout_dct:
                coeffs = self.wdrop(coeffs)  # TODO does this work out of box?
            # round 1
            weights2 = self.to_weights_rectangular(
                coeffs, self.hh_ind_med, self.hh_weights_f_med,
                self.hid_dct_layer_medium, self.hid_dct_layer_medium)
            # round 2
            weights = self.to_weights_rectangular(
                weights2, self.hh_ind_large, self.hh_weights_f_large,
                self.hid_dct_layer, self.hid_dct_layer)

            if w_hh is not None:
                w_hh = torch.cat([w_hh, weights], dim=0)
            else:
                w_hh = weights

        # concatenate both
        weights = torch.cat([w_ih, w_hh], dim=1)
        return weights

    def forward(self, input_, hidden=None, device='cuda'):
        # input shape: (len, B, dim)
        # output shape: (len * B, num_classes)
        outputs = []

        weights = self.get_weights(device)
        if self.weight_drop > 0.0:
            weights = self.wdrop(weights)

        for x in torch.unbind(input_, dim=0):
            h = self.forward_step(x, hidden, weights)
            outputs.append(h[0].clone())
            hidden = h[1]
        op = torch.squeeze(torch.stack(outputs))
        return op, hidden

    def forward_step(self, x, prev_state, weights=None, device='cpu'):
        assert weights is not None
        # One time step forwarding.
        # input x: (B,  in_dim)
        # prev_state: tuple 2 * (B, out_dim)

        if prev_state is None:
            h = torch.zeros([x.size()[0], self.hidden_dim], device=device)
            c = torch.zeros([x.size()[0], self.hidden_dim], device=device)
        else:
            h, c = torch.squeeze(prev_state[0]), torch.squeeze(prev_state[1])

        out = torch.cat([x, h], dim=1)
        out = F.linear(out, weights, self.bias)
        h_out, i_out, f_out, o_out = torch.split(
            out,
            [self.hidden_dim, self.hidden_dim, self.hidden_dim,
             self.hidden_dim],
            dim=1)

        h_out = torch.tanh(h_out)
        i_out = torch.sigmoid(i_out)
        f_out = torch.sigmoid(f_out)
        o_out = torch.sigmoid(o_out)

        if c is None:
            c = i_out * h_out
        else:
            c = i_out * h_out + f_out * c
        out = o_out * torch.tanh(c)
        return out, (out, c)


if __name__ == '__main__':
    # Simple forwarding

    batch_size = 3
    seq_len = 5

    input_dim = 10
    hidden_dim = 20

    sparsity_ih = 0.8
    sparsity_hh = 0.8

    dct_lstm = DctLSTM(input_dim, hidden_dim, sparsity_ih, sparsity_hh)

    input = torch.randn(seq_len, batch_size, input_dim)
    output, (h_, c_) = dct_lstm(input)

    print(output.shape)
