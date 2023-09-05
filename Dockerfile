FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
COPY sources.list /etc/apt/sources.list
COPY .condarc /root/.condarc
WORKDIR /workspace