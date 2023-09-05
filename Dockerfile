FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace
ENV PATH="${PATH}:/home/user/.local/bin"

ARG USER_ID
ARG GROUP_ID

RUN groupadd --gid $USER_ID user
RUN adduser  --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user



RUN apt update \
    && apt install -y \
    && rm -rf /var/lib/apt/lists/*

# Install system packages
USER root
RUN apt-get update --fix-missing && \
  apt-get install -y \
  wget \
  libgtk2.0-dev \
  bzip2 \
  ca-certificates \
  curl \
  git \
  vim \
  g++ \
  gcc \
  zip \
  lrzip \
  unzip \
  graphviz \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libglib2.0-0 \
  libgl1-mesa-glx \
  libhdf5-dev \
  openmpi-bin \
  && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
USER user

# Pip installs
RUN pip install setuptools \
    jupyterlab==2.1.4 \
    notebook==6.0.3 \
    scikit-learn \
    lmdb \
    h5py \
    scipy \
    ipywidgets==7.5.1 \
    tensorboardX \
    openpyxl \
    pandas \
    xlrd



# Other installs
RUN pip install top2vec
RUN pip install seaborn 
RUN pip install tensorboard
RUN pip install markupsafe
RUN pip install matplotlib
RUN pip install yellowbrick
RUN pip install umap-learn
RUN pip install umap-learn[plot]
RUN pip install snakemake
RUN pip install scikit-learn
RUN pip install xgboost 
RUN pip install hyperopt 
RUN pip install nltk 
RUN pip install gensim 

# Jupyter notebook configuration
RUN pip install jupyter ipython ipython[notebook]
RUN pip install jupyter-server
RUN pip install yapf==0.30.0
RUN pip install jupyter_contrib_nbextensions==0.5.1
RUN pip install jupyter_highlight_selected_word==0.2.0
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension install https://github.com/jfbercher/code_prettify/archive/master.zip --user
RUN jupyter nbextension enable code_prettify-master/code_prettify
RUN jupyter nbextension install --py jupyter_highlight_selected_word --user
RUN jupyter nbextension enable highlight_selected_word/main

EXPOSE 8080 8888 6006
