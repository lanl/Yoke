# docker buildx build --progress=plain -f Dockerfile . -t yoke
FROM condaforge/miniforge3:latest

RUN conda install -c conda-forge flit

WORKDIR /workspace
COPY . /workspace

RUN pip3 install torch torchvision

RUN FLIT_ROOT_INSTALL=1 flit install

SHELL ["/bin/bash", "-c"]
# docker run -it yoke /bin/bash
