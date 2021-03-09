FROM ubuntu:18.04

# build-time variables to be set by the Makefile
ARG group
ARG gid
ARG user
ARG uid
#ARG giturl

RUN groupadd --gid $gid $group && \
    useradd --create-home --shell /bin/bash --gid $gid --uid $uid $user && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        less \
        libfreetype6-dev \
        libzmq3-dev \
        nano \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        wget \
        libhdf5-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/miniconda3/bin:$PATH" \
    TINI_VERSION=v0.6.0

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh && \
    HOME=/opt /bin/bash Miniconda3-py38_4.9.2-Linux-x86_64.sh -b && \
    rm Miniconda3-py38_4.9.2-Linux-x86_64.sh && \
    conda update conda && \
    # conda update anaconda && \
    conda update --all && \
    conda install \
        numpy \
        pandas \
        pyyaml \
        scikit-learn \
        setuptools \
        hdf5 \
        && \
    conda env create -f environment.yml \
    chown -R $user:$group /home/$user && \
    chmod +x /usr/bin/tini

Add 
ADD .code /home/$user/code
RUN chown $user:$group -R /home/$user/code
WORKDIR /home/$user/code
# Remove useless remote
RUN git remote | xargs -n1 git remote remove
# And restore the right one
#RUN git remote add origin $giturl
USER $user
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]