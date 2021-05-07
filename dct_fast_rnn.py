# Fast RNN models with DCT-parameterized weights;
# DCT coefficients are parameterised by LSTMs.

import math

import torch
import torch.nn as nn
import torch_dct as dct

from external_torch_dct import DCTLayer
from custom_layer import LinearWithDCT


# Fast weight RNN layer with DCT-parameterized weights;
# DCT coefficients of both feed-forward and recurrent weights are
# parameterised by a "single" LSTM.
class FastDctRNN(nn.Module):
    '''RNN with weights genereted by DCT related ops.'''

    def __init__(self, input_dim, hidden_dim, sparsity_ih, sparsity_hh,
                 fast_weight_drop=0.0, dropout_dct=False, cuda=True,
                 batch_size=-1, coef_scale=True):
        super(FastDctRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_ih = sparsity_ih
        self.sparsity_hh = sparsity_hh
        self.weight_drop = fast_weight_drop
        self.dropout_dct = dropout_dct
        self.cuda = cuda
        self.batch_size = batch_size

        if fast_weight_drop > 0.0:
            self.wdrop = nn.Dropout(fast_weight_drop)

        in_coeffs_dim, in_num_diags = self.get_sparse_config(
            input_dim, hidden_dim, sparsity_ih)

        hidden_coeffs_dim, hidden_num_diags = self.get_sparse_config(
            hidden_dim, hidden_dim, sparsity_hh)

        self.in_dct_layer = DCTLayer(
            in_features=input_dim, type='dct', norm='ortho', cuda=cuda)
        self.hid_dct_layer = DCTLayer(
            in_features=hidden_dim, type='dct', norm='ortho', cuda=cuda)
        self.in_idct_layer = DCTLayer(
            in_features=input_dim, type='idct', norm='ortho', cuda=cuda)
        self.hid_idct_layer = DCTLayer(
            in_features=hidden_dim, type='idct', norm='ortho', cuda=cuda)

        self.linear_with_dct = LinearWithDCT.apply
        # number of diagonals
        self.in_num_diags = in_num_diags
        self.hidden_num_diags = hidden_num_diags

        # number of DCT coefficients
        self.in_coeffs_dim = in_coeffs_dim
        self.hidden_coeffs_dim = hidden_coeffs_dim

        self.fast_h_dim = self.in_coeffs_dim + self.hidden_coeffs_dim
        self.fast_weights_lstm = nn.LSTM(input_dim, self.fast_h_dim)
        print(f"Number of fast parameters: {self.fast_h_dim}")

        if cuda:
            device = "cuda"

        self.coef_scale = coef_scale
        if coef_scale:
            print(f"coef_scale: {coef_scale}")
            self.coef_scaler = nn.Parameter(
                torch.ones([self.fast_h_dim], device=device),
                requires_grad=True)

        # This assumes that the batch size is same for all batches;
        # which is ensured by the batch construction, but still not nice.
        # TODO make this flexible.
        self.ih_weights_f = torch.zeros(
            [batch_size, self.hidden_dim, self.input_dim], device=device)
        self.hh_weights_f = torch.zeros(
            [batch_size, self.hidden_dim, self.hidden_dim], device=device)

        # ih
        list = []
        # shape (2, len_coeffs)
        ind = torch.triu_indices(
            self.hidden_dim, self.input_dim, self.in_num_diags, device=device)
        for i in range(batch_size):
            for t in torch.unbind(ind, 1):
                # (2, len_coeffs) -> (3, len_coeffs)
                list.append(
                    torch.cat((torch.tensor([i], device=device), t), dim=0))
        self.ih_ind = torch.stack(list).t()
        # hh
        list = []
        ind = torch.triu_indices(self.hidden_dim, self.hidden_dim,
                                 self.hidden_num_diags, device=device)

        for i in range(batch_size):
            for t in torch.unbind(ind, 1):
                list.append(
                    torch.cat((torch.tensor([i], device=device), t), dim=0))
        self.hh_ind = torch.stack(list).t()

        bias_init = torch.rand([hidden_dim])
        initrange = 1.0 / math.sqrt(hidden_dim)
        nn.init.uniform_(bias_init, -initrange, initrange)
        self.bias = nn.Parameter(bias_init)

    def get_dct_init(self, len_coeffs, dim_out, dim_in, diag_shift):

        factor = 1.
        init = torch.rand([dim_out, dim_in])
        if self.cuda:  # TODO update to device.
            init = init.cuda()

        initrange = 1.0 / math.sqrt(dim_out)
        nn.init.uniform_(init, -initrange, initrange)
        init_f = torch.fliplr(dct.dct_2d(init, norm='ortho'))
        ind = torch.triu_indices(dim_out, dim_in, diag_shift)
        coeffs_init = init_f[tuple(ind)] * factor

        return coeffs_init

    def to_weights(self, coeffs, ind, zero_weights, linear1, linear2):

        zero_weights_ = zero_weights.clone()
        weights = torch.fliplr(zero_weights_.index_put_(tuple(ind), coeffs))
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
        coeffs = self.coeffs_ih

        if self.dropout_dct:
            coeffs = self.wdrop(coeffs)
        weights = self.to_weights(
            coeffs, self.ih_ind, self.ih_weights_f,
            self.in_dct_layer, self.hid_dct_layer)
        if w_ih is not None:
            w_ih = torch.cat([w_ih, weights], dim=0)
        else:
            w_ih = weights

        # hidden to hidden
        w_hh = None
        coeffs = self.coeffs_hh
        if self.dropout_dct:
            coeffs = self.wdrop(coeffs)
        weights = self.to_weights(
            coeffs, self.hh_ind, self.hh_weights_f,
            self.hid_dct_layer, self.hid_dct_layer)

        if w_hh is not None:
            w_hh = torch.cat([w_hh, weights], dim=0)
        else:
            w_hh = weights

        # concatenate both
        # weights = torch.cat([w_ih, w_hh], dim=1)
        return (w_ih, w_hh)

    def forward(self, input_, hidden=None):
        # input shape: (len, B, dim)
        # output shape: (len * B, num_classes)
        outputs = []

        if hidden is None:
            hidden_fast_weight = (
                torch.zeros(1, input_.shape[1], self.fast_h_dim,
                            device=input_.device),
                torch.zeros(1, input_.shape[1], self.fast_h_dim,
                            device=input_.device))
            hidden = torch.zeros(
                1, input_.shape[1], self.hidden_dim, device=input_.device)
        else:
            h, hidden_fast_weight = hidden
            hidden = h

        # compute fast weight first.
        fast_output, hidden_fast_weight = self.fast_weights_lstm(
            input_, hidden_fast_weight)

        fast_output = torch.unbind(fast_output, dim=0)

        for i, x in enumerate(torch.unbind(input_, dim=0)):
            weights = fast_output[i]
            if self.weight_drop > 0.0:
                weights = self.wdrop(weights)
            h = self.forward_step(x, hidden, weights)
            outputs.append(h.clone())
            hidden = h
        op = torch.squeeze(torch.stack(outputs))
        hidden = (h, hidden_fast_weight)
        return op, hidden

    def forward_step(self, x, prev_state, weights=None):
        assert weights is not None
        # One time step forwarding.
        # input x: (B,  in_dim)

        # apply scalers to coeffs:
        if self.coef_scale:
            weights = self.coef_scaler.unsqueeze(0) * weights

        ih_weight, hh_weight = torch.split(
            weights, [self.in_coeffs_dim, self.hidden_coeffs_dim], dim=1)
        h = torch.squeeze(prev_state)

        bsz = x.shape[0]
        if bsz != self.batch_size:  # take sub-tensors
            total_dim_coeffs = int(
                bsz * self.ih_ind.shape[-1] / self.batch_size)
            ih_ind = self.ih_ind[:, : total_dim_coeffs]
            total_dim_coeffs = int(
                bsz * self.hh_ind.shape[-1] / self.batch_size)
            hh_ind = self.hh_ind[:, : total_dim_coeffs]
        else:
            ih_ind = self.ih_ind
            hh_ind = self.hh_ind

        out = self.linear_with_dct(
            x, ih_weight,
            self.in_idct_layer.weight,
            self.hid_idct_layer.weight,
            self.in_dct_layer.weight, self.hid_dct_layer.weight,
            ih_ind, self.ih_weights_f, None)

        out = out + self.linear_with_dct(
            h, hh_weight, self.hid_idct_layer.weight,
            self.hid_idct_layer.weight, self.hid_dct_layer.weight,
            self.hid_dct_layer.weight,
            hh_ind, self.hh_weights_f, self.bias)

        out = torch.tanh(out)

        return out


# Fast weight RNN layer with DCT-parameterized weights;
# DCT coefficients of feed-forward and recurrent weights are
# parameterised by "separate" LSTMs.
class SeparateFastDctRNN(nn.Module):
    '''RNN with weights genereted by DCT related ops.'''

    def __init__(self, input_dim, hidden_dim, sparsity_ih, sparsity_hh,
                 fast_weight_drop=0.0, dropout_dct=False, cuda=True,
                 batch_size=-1, coef_scale=True):
        super(SeparateFastDctRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_ih = sparsity_ih
        self.sparsity_hh = sparsity_hh
        self.weight_drop = fast_weight_drop
        self.dropout_dct = dropout_dct
        self.cuda = cuda
        self.batch_size = batch_size

        if fast_weight_drop > 0.0:
            self.wdrop = nn.Dropout(fast_weight_drop)

        in_coeffs_dim, in_num_diags = self.get_sparse_config(
            input_dim, hidden_dim, sparsity_ih)

        hidden_coeffs_dim, hidden_num_diags = self.get_sparse_config(
            hidden_dim, hidden_dim, sparsity_hh)

        self.in_dct_layer = DCTLayer(
            in_features=input_dim, type='dct', norm='ortho', cuda=cuda)
        self.hid_dct_layer = DCTLayer(
            in_features=hidden_dim, type='dct', norm='ortho', cuda=cuda)
        self.in_idct_layer = DCTLayer(
            in_features=input_dim, type='idct', norm='ortho', cuda=cuda)
        self.hid_idct_layer = DCTLayer(
            in_features=hidden_dim, type='idct', norm='ortho', cuda=cuda)

        self.linear_with_dct = LinearWithDCT.apply
        # number of diagonals
        self.in_num_diags = in_num_diags
        self.hidden_num_diags = hidden_num_diags

        # number of coefficients
        self.in_coeffs_dim = in_coeffs_dim
        self.hidden_coeffs_dim = hidden_coeffs_dim

        self.fast_weights_lstm_ih = nn.LSTM(input_dim, in_coeffs_dim)
        print(f"Number of fast params input-to-hidden: {in_coeffs_dim}")

        self.fast_weights_lstm_hh = nn.LSTM(input_dim, hidden_coeffs_dim)
        print(f"Number of fast params hidden-to-hidden: {hidden_coeffs_dim}")

        if cuda:
            device = "cuda"

        self.coef_scale = coef_scale
        if coef_scale:
            print(f"coef_scale: {coef_scale}")
            self.coef_scaler_ih = nn.Parameter(
                torch.ones([in_coeffs_dim], device=device), requires_grad=True)
            self.coef_scaler_hh = nn.Parameter(
                torch.ones([hidden_coeffs_dim], device=device),
                requires_grad=True)

        # This assumes that the batch size is same for all batches;
        # which is ensured by the batch construction, but still not nice.
        # TODO make this flexible.
        self.ih_weights_f = torch.zeros(
            [batch_size, self.hidden_dim, self.input_dim], device=device)
        self.hh_weights_f = torch.zeros(
            [batch_size, self.hidden_dim, self.hidden_dim], device=device)

        # ih
        list = []
        # shape (2, len_coeffs)
        ind = torch.triu_indices(
            self.hidden_dim, self.input_dim, self.in_num_diags, device=device)
        for i in range(batch_size):
            # (2, len_coeffs) -> (3, len_coeffs)
            for t in torch.unbind(ind, 1):
                list.append(
                    torch.cat((torch.tensor([i], device=device), t), dim=0))
        self.ih_ind = torch.stack(list).t()
        # hh
        list = []
        ind = torch.triu_indices(
            self.hidden_dim, self.hidden_dim, self.hidden_num_diags,
            device=device)

        for i in range(batch_size):
            for t in torch.unbind(ind, 1):
                list.append(
                    torch.cat((torch.tensor([i], device=device), t), dim=0))
        self.hh_ind = torch.stack(list).t()

        bias_init = torch.rand([hidden_dim])
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
        # coeffs_init = init_f[ind.numpy()] * factor
        coeffs_init = init_f[tuple(ind)] * factor

        return coeffs_init

    def to_weights(self, coeffs, ind, zero_weights, linear1, linear2):

        zero_weights_ = zero_weights.clone()
        weights = torch.fliplr(zero_weights_.index_put_(tuple(ind), coeffs))
        # weights = dct.idct_2d(weights)
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
        coeffs = self.coeffs_ih

        if self.dropout_dct:
            coeffs = self.wdrop(coeffs)
        weights = self.to_weights(
            coeffs, self.ih_ind, self.ih_weights_f,
            self.in_dct_layer, self.hid_dct_layer)
        if w_ih is not None:
            w_ih = torch.cat([w_ih, weights], dim=0)
        else:
            w_ih = weights

        # hidden to hidden
        w_hh = None
        coeffs = self.coeffs_hh
        if self.dropout_dct:
            coeffs = self.wdrop(coeffs)
        weights = self.to_weights(
            coeffs, self.hh_ind, self.hh_weights_f,
            self.hid_dct_layer, self.hid_dct_layer)

        if w_hh is not None:
            w_hh = torch.cat([w_hh, weights], dim=0)
        else:
            w_hh = weights

        # concatenate both
        # weights = torch.cat([w_ih, w_hh], dim=1)
        return (w_ih, w_hh)

    def forward(self, input_, hidden=None, device='cuda'):
        # input shape: (len, B, dim)
        # output shape: (len * B, num_classes)
        outputs = []

        if hidden is None:
            hidden_fast_weight_ih = (
                torch.zeros(1, input_.shape[1], self.in_coeffs_dim,
                            device=input_.device),
                torch.zeros(1, input_.shape[1], self.in_coeffs_dim,
                            device=input_.device))
            hidden_fast_weight_hh = (
                torch.zeros(1, input_.shape[1], self.hidden_coeffs_dim,
                            device=input_.device),
                torch.zeros(1, input_.shape[1], self.hidden_coeffs_dim,
                            device=input_.device))
            hidden = torch.zeros(
                1, input_.shape[1], self.hidden_dim, device=input_.device)
        else:
            h, hidden_fast_weight_ih, hidden_fast_weight_hh = hidden
            hidden = h

        # compute fast weight first.
        fast_output_ih, hidden_fast_weight_ih = self.fast_weights_lstm_ih(
            input_, hidden_fast_weight_ih)
        fast_output_hh, hidden_fast_weight_hh = self.fast_weights_lstm_hh(
            input_, hidden_fast_weight_hh)

        fast_output_ih = torch.unbind(fast_output_ih, dim=0)
        fast_output_hh = torch.unbind(fast_output_hh, dim=0)

        for i, x in enumerate(torch.unbind(input_, dim=0)):
            weights_ih = fast_output_ih[i]
            weights_hh = fast_output_hh[i]

            if self.weight_drop > 0.0:
                weights_ih = self.wdrop(weights_ih)
                weights_hh = self.wdrop(weights_hh)

            h = self.forward_step(x, hidden, weights_ih, weights_hh)
            outputs.append(h.clone())
            hidden = h
        op = torch.squeeze(torch.stack(outputs))
        hidden = (h, hidden_fast_weight_ih, hidden_fast_weight_hh)
        return op, hidden

    def forward_step(self, x, prev_state, ih_weight=None, hh_weight=None):
        assert ih_weight is not None
        assert hh_weight is not None
        # One time step forwarding.
        # input x: (B,  in_dim)
        # prev_state: tuple 2 * (B, out_dim)

        h = torch.squeeze(prev_state)

        bsz = x.shape[0]
        if bsz != self.batch_size:  # take sub-tensors
            total_dim_coeffs = int(
                bsz * self.ih_ind.shape[-1] / self.batch_size)
            ih_ind = self.ih_ind[:, : total_dim_coeffs]
            total_dim_coeffs = int(
                bsz * self.hh_ind.shape[-1] / self.batch_size)
            hh_ind = self.hh_ind[:, : total_dim_coeffs]
        else:
            ih_ind = self.ih_ind
            hh_ind = self.hh_ind

        if self.coef_scale:
            ih_weight = ih_weight * self.coef_scaler_ih.unsqueeze(0)
            hh_weight = hh_weight * self.coef_scaler_hh.unsqueeze(0)

        out = self.linear_with_dct(
            x, ih_weight,
            self.in_idct_layer.weight, self.hid_idct_layer.weight,
            self.in_dct_layer.weight, self.hid_dct_layer.weight,
            ih_ind, self.ih_weights_f, None)

        out = out + self.linear_with_dct(
            h, hh_weight, self.hid_idct_layer.weight,
            self.hid_idct_layer.weight, self.hid_dct_layer.weight,
            self.hid_dct_layer.weight,
            hh_ind, self.hh_weights_f, self.bias)

        out = torch.tanh(out)

        return out


if __name__ == '__main__':
    # Simple forwarding

    batch_size = 3
    seq_len = 5

    input_dim = 10
    hidden_dim = 20

    sparsity_ih = 0.8
    sparsity_hh = 0.8

    print('FastDctRNN')
    dct_fast_rnn = FastDctRNN(
        input_dim, hidden_dim, sparsity_ih, sparsity_hh, batch_size=batch_size)

    dct_fast_rnn = dct_fast_rnn.to('cuda')

    input = torch.randn(seq_len, batch_size, input_dim, device='cuda')
    output, all_states = dct_fast_rnn(input)

    print(output.shape)

    print('SeparateFastDctRNN')
    dct_fast_rnn_twin = SeparateFastDctRNN(
        input_dim, hidden_dim, sparsity_ih, sparsity_hh, batch_size=batch_size)

    dct_fast_rnn_twin = dct_fast_rnn_twin.to('cuda')

    output, all_states = dct_fast_rnn_twin(input)

    print(output.shape)
