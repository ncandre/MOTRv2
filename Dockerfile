# Using suricate as a base.
ARG py_version=3.7.3
# Staying with cuda 10.1 as we're using the recommended pytorch version: 1.5
# ARG cuda_cudnn_version=10.1-cudnn7-devel
ARG cuda_cudnn_version=11.2.0-cudnn8-devel
FROM 766281746212.dkr.ecr.eu-west-1.amazonaws.com/skynet/suricate:latest_prod_${cuda_cudnn_version}_3.7.3 as base
# FROM nvidia/cuda:11.1.1-runtime-ubuntu18.04 as base

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache --mount=type=cache,target=/var/lib/apt,id=apt-lib \
    apt-get -y update \
    && apt-get install -y --force-yes curl git python3-pip python3-venv ssh \
    && mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts


FROM base as multi-stage-linux-engine-core
ARG py_version
ARG cuda_cudnn_version

WORKDIR /skynet/libs/engine-core

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache --mount=type=cache,target=/var/lib/apt,id=apt-lib \
    apt-get -y update \
    && apt-get install -y --force-yes libpq-dev python3-pip \
    && rm -rf /tmp/*

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache --mount=type=cache,target=/var/lib/apt,id=apt-lib \
    apt-get -y update \
    && apt-get install -y --force-yes unzip \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip && ./aws/install && rm -rf aws*


FROM multi-stage-linux-engine-core as tmp-step

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache --mount=type=cache,target=/var/lib/apt,id=apt-lib \
    apt-get -y update \
    && apt-get install -y --force-yes python3-pip \
    && rm -rf /tmp/*

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache,id=user-cache \
    --mount=src=./requirements.txt,target=/MOTRv2/requirements.txt \
    pip install --no-cache-dir --upgrade pip && \
    pip install "setuptools>=58.1.0,<60.*" "wheel>=0.37.0" && \
    pip install -r /MOTRv2/requirements.txt


FROM tmp-step as final-form

ARG py_version
ARG cuda_cudnn_version
WORKDIR /MOTRv2

# RUN --mount=type=ssh \
#     --mount=type=cache,target=/root/.cache,id=user-cache \
#     pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html && \
#     pip install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'
RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache,id=user-cache \
    pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache --mount=type=cache,target=/var/lib/apt,id=apt-lib \
    apt-get -y update \
    && apt-get install -y --force-yes neovim nano zsh git jq unzip ca-certificates curl gnupg lsb-release \
    && rm -rf /tmp/* \
    # Install docker
    && curl -fsSL https://get.docker.com -o get-docker.sh && sh ./get-docker.sh && rm ./get-docker.sh && \
    pip install docker-compose && \
    # Install AWS CLI
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip -o awscliv2.zip && ./aws/install --update && rm -rf aws* && \
    # Install and setup OhMyZSH
    sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    chsh -s ~/.zshrc && \
    sed -i 's/plugins=(git)/plugins=(git copybuffer dirhistory)/g' ~/.zshrc && \
    sed -i -- 's/\(ZSH_THEME=\)"robbyrussell"/\1"maran"/' ~/.zshrc && \
    echo 'alias nvim=vim' >> ~/.zshrc && \
    echo 'PATH=$HOME/.poetry/bin:$PATH' >> ~/.zshrc && \
    # Allow to save zsh history when we quit the docker
    echo "setopt nohistsavebycopy" >> ~/.zshrc


# Need a GPU to run
# Current hack is to stop docker creation before, launch a container, make the install and save the container as a new image.
# RUN --mount=type=ssh \
#     --mount=type=cache,target=/root/.cache,id=user-cache \
#     --mount=src=./,target=/MOTRv2 \
#     python /MOTRv2/motmodels/ops/setup.py build --build-base=motmodels/ops/ install

ENTRYPOINT ["zsh"]
