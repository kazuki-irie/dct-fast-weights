# Taken from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
#
# (c) Copyright 2018 Ziyang Hu.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn

import torch_dct as dct


class DCTLayer(nn.Linear):
    """Implement any DCT as a linear layer.

    In practice this executes around 50x faster on GPU.
    Unfortunately, the DCT matrix is stored, which will increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use.
    """

    def __init__(self, in_features, type, norm=None, bias=False, cuda=True):
        self.type = type
        self.N = in_features
        self.norm = norm
        self.cuda = cuda
        super(DCTLayer, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.cuda:
            I = I.cuda()
        if self.type == 'dct1':
            self.weight.data = dct.dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = dct.idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct.dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = dct.idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False  # don't learn this!
