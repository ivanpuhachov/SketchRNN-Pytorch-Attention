# SketchRNN Pytorch
SketchRNN is a seq2seq VAE model which draws pictures.  

Paper: https://arxiv.org/abs/1704.03477

> This repository contatins the original implementation + model with attention in decoder

## Related works
 * "SketchEmbedNet: Learning Novel Concepts by Imitating Drawings" https://arxiv.org/pdf/2009.04806.pdf
 * "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" https://arxiv.org/abs/2006.16236
 * "Sketch-BERT: Learning Sketch Bidirectional Encoder Representation from Transformers by Self-supervised Learning of Sketch Gestalt" https://arxiv.org/abs/2005.09159
 * "Creative Sketch Generation" https://arxiv.org/abs/2011.10039

## DoodlerGAN data (Creative Birds)
 * Github repo: https://github.com/facebookresearch/DoodlerGAN
 * Google Drive files: https://drive.google.com/drive/folders/14ZywlSE-khagmSz23KKFbLCQLoMOxPzl

### Simplification
Requirements: `pip install simplification` (see https://pypi.org/project/simplification/)

***

## Tips for a faster convergence
Source: https://github.com/magenta/magenta/issues/742
- anneal KL faster: use 0.9999 rather than 0.99995
- use learning rate at 0.0001 and not anneal below this rate to 0.00001, at
the higher risk of NaNs
- turn off dropout, train faster but risk overfitting too soon

## This is a fork
Original repo is here https://github.com/OhataKenji/SketchRNN-Pytorch. It provides the backbone (original model and trainer).

Improvements:
 - model with attention
 - data augmentation 
 - experiment logs, validation

Cudos to creator.




