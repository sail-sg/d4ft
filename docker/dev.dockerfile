# bring in the micromamba image so we can copy files from it
FROM mambaorg/micromamba:1.5.8 as micromamba

# This is the image we are going add micromaba to:
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update \
    && apt-get install -y python3-pip python3-dev golang git wget curl zsh tmux vim
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

ARG HOME=/root
ENV GOBIN=/usr/local/bin
ENV PATH=$PATH:${GOBIN}
WORKDIR $HOME

RUN go install github.com/bazelbuild/bazelisk@v1.19.0 && ln -sf /usr/local/bin/bazelisk /usr/local/bin/bazel
RUN go install github.com/bazelbuild/buildtools/buildifier@latest
RUN bazel version

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/lib/go/bin/go /usr/bin/go

RUN apt-get update \
    && apt-get install -y clang-format clang-tidy swig qtdeclarative5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY apt_install.txt .
RUN apt-get update
RUN apt-get install -y `cat apt_install.txt`

RUN echo 'export SHELL=/bin/zsh' >> ~/.bash_profile
RUN echo 'exec /bin/zsh -l' >> ~/.bash_profile

USER root

ARG MAMBA_USER=mambauser
ARG MAMBA_USER_ID=57439
ARG MAMBA_USER_GID=57439
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
ENV USER=$MAMBA_USER

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

USER root

# NOTE: ml_collections does not work with python 3.12 yet
RUN micromamba install --yes --name base --channel conda-forge -c conda-forge/label/libint_dev \
    python=3.11 numpy scipy fftw 'gxx<12' psi4

ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

RUN export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH

# Switch to non-root user for the final image
USER $MAMBA_USER

