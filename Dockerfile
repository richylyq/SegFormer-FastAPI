# pull official base image
FROM nvidia/cuda:10.1-cudnn8-runtime-ubuntu18.04

# set environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# ENV WGETRC=/usr/.wgetrc

# set work directory
WORKDIR /usr/src

# set up directories
COPY ./src/checkpoints /usr/src/checkpoints

# copy scripts
COPY ./src/requirements.txt /usr/src/requirements.txt
COPY ./src/download_weights.sh /usr/src/download_weights.sh
# COPY ./.wgetrc /usr/.wgetrc

# install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx wget git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux \
    && pip3 install --upgrade pip setuptools wheel \
    && pip3 install -r requirements.txt \
    && rm -rf /root/.cache/pip

# download sample data and weights
RUN ["chmod", "+x", "./download_weights.sh"]
RUN ./download_weights.sh
RUN rm -rf /usr/.wgetrc

# copy source
COPY ./src /usr/src

# set work directory
WORKDIR /usr/src

ENTRYPOINT ["bash", "entrypoint.sh"]
