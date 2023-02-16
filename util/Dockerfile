# This generates the docker image to run stonne without python frontend

FROM ubuntu:18.04
CMD ["bash"]
LABEL maintainer=hansikam94@gmail.com
RUN apt-get update && apt-get install  -y g++ software-properties-common  \
    && add-apt-repository ppa:ubuntu-toolchain-r/test  \
    && apt update  && apt install -y gcc-9  \
    && apt-get install -y libstdc++6 curl wget htop  \
    && apt-get install -y cmake  && apt-get install -y vim  \
    && apt-get install -y bc && wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh -O ~/miniconda.sh  \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda  && rm ~/miniconda.sh  && apt-get install -y git && echo "Dependancies installed for Stonne"
ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN conda init bash
RUN git clone https://github.com/hansikaweerasena/stonne.git /home/stonne
#WORKDIR /home/stonne/pytorch-frontend
#RUN export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#RUN python setup.py install
#WORKDIR /home/stonne/pytorch-frontend/stonne_connection/
#RUN python setup.py install
#RUN conda install torchvision=0.8.1
#RUN python setup.py install
WORKDIR /home/stonne/stonne
RUN make

