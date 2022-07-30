FROM continuumio/miniconda3 

WORKDIR /root/
COPY . .

# Create and activate conda environment
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "drnld", "/bin/bash", "-c"]
