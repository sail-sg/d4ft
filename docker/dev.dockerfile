FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ENV PATH=/miniconda/bin:${PATH}

# Install dependencies
RUN apt-get update \
    && apt-get install -y python3-pip python3-dev golang-1.18 git wget curl zsh tmux vim \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/lib/go-1.18/bin/go /usr/bin/go
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

ARG HOME=/root
ARG PATH=$PATH:$HOME/go/bin
WORKDIR $HOME
RUN git clone https://github.com/gpakosz/.tmux.git
RUN ln -s -f .tmux/.tmux.conf
RUN cp .tmux/.tmux.conf.local .
RUN echo "set-option -g default-shell /bin/zsh" >> .tmux.conf.local
RUN echo "set-option -g history-limit 10000" >> .tmux.conf.local
RUN echo "export PATH=$PATH:$HOME/go/bin" >> .zshrc

RUN go install github.com/bazelbuild/bazelisk@latest && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel
RUN go install github.com/bazelbuild/buildtools/buildifier@latest
RUN $HOME/go/bin/bazel version

RUN useradd -ms /bin/zsh github-action

RUN apt-get update \
    && apt-get install -y clang-format clang-tidy swig qtdeclarative5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    conda update -y conda

RUN conda install --quiet --yes python=${PYTHON_VERSION} && \
    conda clean --yes --all

# Upgrade pip, install py libs
RUN pip install --upgrade pip

# install D4FT dependencies
WORKDIR /app
COPY third_party/pip_requirements/requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# jax
RUN pip uninstall jax jaxlib -y
RUN pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# zsh
RUN curl -L git.io/antigen > ~/antigen.zsh # install antigen
RUN git clone https://github.com/zsh-users/zsh-autosuggestions $HOME/.zsh/zsh-autosuggestions # plugin
RUN printf "source ~/antigen.zsh \n\
antigen use oh-my-zsh \n\
antigen bundle git \n\
antigen bundle heroku \n\
antigen bundle pip \n\
antigen bundle lein \n\
antigen bundle command-not-found \n\
antigen bundle poetry \n\
antigen bundle zsh-users/zsh-syntax-highlighting \n\
antigen bundle zsh-users/zsh-autosuggestions \n\
source ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh \n\
antigen theme https://github.com/denysdovhan/spaceship-prompt spaceship \n\
antigen apply" > /root/.zshrc
