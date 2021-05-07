import torch


# DCT-parameterized linear layer with custom backward pass
class LinearWithDCT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, coeffs, idct_weight1, idct_weight2, dct_weight1,
                dct_weight2, ind, zero_weights, bias=None):
        ctx.save_for_backward(
            input, coeffs, idct_weight1, idct_weight2,
            dct_weight1, dct_weight2, ind, zero_weights, bias)

        weight = coeffs_to_weight(
            coeffs, idct_weight1, idct_weight2, ind, zero_weights)

        output = torch.bmm(
            input.view(coeffs.shape[0], -1, idct_weight1.shape[0]),
            weight).squeeze()

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input, coeffs, idct_weight1, idct_weight2, dct_weight1, dct_weight2,
         ind, zero_weights, bias) = ctx.saved_tensors
        grad_input = grad_coeffs = grad_bias = None

        # recompute weights.
        weight = coeffs_to_weight(
            coeffs, idct_weight1, idct_weight2, ind, zero_weights)

        if ctx.needs_input_grad[0]:
            grad_input = torch.bmm(
                grad_output.view(coeffs.shape[0], -1, idct_weight2.shape[0]),
                weight.transpose(-1, -2)).squeeze()

        if ctx.needs_input_grad[1]:
            # (B, out_dim, in_dim)  # replace einsum, slow.
            grad_weight = torch.einsum('bi,bj->bij', (input, grad_output))

            grad_coeffs = weight_to_coeffs_grad(
                grad_weight, coeffs, dct_weight1, dct_weight2, ind)

        if bias is not None and ctx.needs_input_grad[8]:
            grad_bias = grad_output.sum(0)

        return (grad_input, grad_coeffs, None, None, None, None, None, None,
                grad_bias)


def coeffs_to_weight(coeffs, dct_mat1, dct_mat2, ind, zero_weights_):
    # coeffs shape: (B, len_g)
    # Matrices for *inverse* DCT

    # generate top-left indices and sparse canvas
    zero_weights = zero_weights_[:coeffs.shape[0], :, :].clone()

    weights = zero_weights.index_put_(
        tuple(ind), coeffs.reshape([coeffs.shape[0] * coeffs.shape[1]]))

    weights = torch.flip(weights, [2])
    weights = torch.matmul(dct_mat2, weights)
    weights = torch.matmul(dct_mat1, weights.transpose(-1, -2))

    return weights


def weight_to_coeffs_grad(grad_weight, coeffs, dct_mat1, dct_mat2, ind):
    # coeffs shape: (B, len_g)
    bsz, g_dim = coeffs.shape

    # Apply DCT to weight gradient.
    grad_weight = torch.matmul(dct_mat1, grad_weight)
    grad_weight = torch.matmul(
        dct_mat2, grad_weight.transpose(-1, -2))

    # flip back before
    grad_weight = torch.flip(grad_weight, [2])
    grad_weights = grad_weight[tuple(ind)]

    return grad_weights.reshape([bsz, g_dim])


if __name__ == '__main__':
    # Run a gradient test.
    from torch.autograd import gradcheck
    from external_torch_dct import DCTLayer

    print(f"Using torch version: {torch.__version__}")

    def get_sparse_config(in_dim, out_dim, sparsity_level):
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

    in_dim = 9
    out_dim = 11
    batch_size = 3
    device = 'cuda'

    coeffs_len, num_diag = get_sparse_config(
        in_dim, out_dim, sparsity_level=0.8)

    in_idct_layer = DCTLayer(
        in_features=in_dim, type='idct', norm='ortho').to(device)
    hid_idct_layer = DCTLayer(
        in_features=out_dim, type='idct', norm='ortho').to(device)

    in_dct_layer = DCTLayer(
        in_features=in_dim, type='dct', norm='ortho').to(device)
    hid_dct_layer = DCTLayer(
        in_features=out_dim, type='dct', norm='ortho').to(device)

    idct_weight1 = in_idct_layer.weight.double()
    idct_weight2 = hid_idct_layer.weight.double()

    dct_weight1 = in_dct_layer.weight.double()
    dct_weight2 = hid_dct_layer.weight.double()

    bias = torch.rand(
        [out_dim], requires_grad=True, dtype=torch.double, device=device)

    coeffs = torch.rand(
        [batch_size, coeffs_len], requires_grad=True, dtype=torch.double,
        device=device)

    input = torch.rand(
        [batch_size, in_dim], requires_grad=True, dtype=torch.double,
        device=device)

    zero_weights = torch.zeros(
        [batch_size, out_dim, in_dim], dtype=torch.double, device=device)

    list = []
    ind = torch.triu_indices(out_dim, in_dim, num_diag, device=device)
    for i in range(batch_size):
        for t in torch.unbind(ind, 1):  # (2, len_coeffs) -> (3, len_coeffs)
            list.append(
                torch.cat((torch.tensor([i], device=device), t), dim=0))
    ind = torch.stack(list).t()

    linear_lay_with_dct = LinearWithDCT.apply
    input_tuple = (
        input, coeffs, idct_weight1, idct_weight2, dct_weight1, dct_weight2,
        ind, zero_weights, bias)

    test = gradcheck(linear_lay_with_dct, input_tuple, eps=1e-6, atol=1e-4)

    if test:
        print("Gradient test pass!")
