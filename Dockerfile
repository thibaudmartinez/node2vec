FROM python:3.7-slim

# Install the notebook package
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache notebook

# Create user with a home directory
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}

# Install GCC 8
RUN apt-get -y update && \
    apt-get -y install gcc-8 g++-8 \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s $(which gcc-8) /usr/bin/gcc
RUN ln -s $(which g++-8) /usr/bin/g++

# Install node2vec library
COPY . ${HOME}/node2vec
WORKDIR ${HOME}/node2vec
RUN pip install --no-cache pybind11==2.4.3
RUN python setup.py build
RUN pip install .
RUN rm -rf ${HOME}/node2vec

COPY *.ipynb ${HOME}
WORKDIR ${HOME}
RUN chown -R ${NB_UID} ${HOME}
USER ${USER}
