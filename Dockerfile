FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

RUN conda install -y tensorboard matplotlib