FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
COPY sources.list /etc/apt/sources.list
# COPY .condarc /root/.condarc
# RUN apt-get update && apt-get install -y openssh-server
# RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && echo "root:1001" | chpasswd