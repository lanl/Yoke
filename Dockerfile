# docker buildx build --progress=plain -f Dockerfile . -t yoke
# # or with Charliecloud
# ch-image build -f Dockerfile . -t yoke
# ch-convert yoke yoke.sqfs

FROM condaforge/miniforge3:latest

RUN conda install -c conda-forge flit

WORKDIR /workspace
COPY . /workspace

RUN pip3 install torch==2.11.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128 
# # use below for CPU build:
# RUN pip3 install torch torchvision --resume-retries 2 --index-url https://download.pytorch.org/whl/cpu

ENV FLIT_ROOT_INSTALL=1
RUN flit install --symlink --deps develop

SHELL ["/bin/bash", "-c"]

# docker run -it yoke /bin/bash
# # or with Charliecloud
# ch-run -d -W --unset-env="*" --set-env --bind $PWD:/mnt/workspace --cd /mnt/workspace yoke.sqfs -- /bin/bash
