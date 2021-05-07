## DCT based fast weights

This repository contains the official code for the paper:
[Training and Generating Neural Networks in Compressed Weight Space](https://openreview.net/forum?id=qU1EUxdVd_D).

The main code includes:
* DCT LSTM: LSTM whose weights are encoded by discrete cosine transform (DCT).
* DCT Fast weight RNNs: RNNs whose weights are encoded by DCT and the DCT coefficients are parameterized by LSTMs.

The language modeling experiments reported in the paper were produced by porting code (with minor changes due to some clean-up) of this repository in a fork of [this toolkit](https://github.com/manuvn/lpRNN-awd-lstm-lm).


## Requirements

* `torch_dct` (can be installed via `pip install torch_dct`)
* PyTorch with a version compatible with `torch_dct`. Our experiments were conducted using PyTorch version `1.6.0` .
More recent versions are apparently not compatible with torch_dct (at least at the time of writing this file).
We recommend to run `python custom_layer.py` to check the compatibility.

## References
If you make use of this toolkit for your experiments, please cite:
```
@inproceedings{irie2021training,
  title={Training and Generating Neural Networks in Compressed Weight Space},
  author={Kazuki Irie and J{\"u}rgen Schmidhuber},
  booktitle={Neural Compression: From Information Theory to Applications -- Workshop @ ICLR 2021},
  year={2021},
  address={Virtual only},
  month=may
}
```
