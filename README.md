# SketchRNN Pytorch

## Tips for a faster convergence
Source: https://github.com/magenta/magenta/issues/742
- anneal KL faster: use 0.9999 rather than 0.99995
- use learning rate at 0.0001 and not anneal below this rate to 0.00001, at
the higher risk of NaNs
- turn off dropout, train faster but risk overfitting too soon

## This is a fork
Original repo is here https://github.com/OhataKenji/SketchRNN-Pytorch.

Cudos to creator.

SketchRNN is a seq2seq VAE model which draws pictures.  
Paper: https://arxiv.org/abs/1704.03477

