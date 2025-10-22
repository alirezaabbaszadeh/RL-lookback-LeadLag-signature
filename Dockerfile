# Simple Dockerfile for reproducible runs
FROM condaforge/miniforge3:latest

WORKDIR /workspace

# Copy environment and requirements first to leverage Docker layer cache
COPY environment.yml /workspace/environment.yml
COPY requirements.txt /workspace/requirements.txt

# Create base conda env and install Python deps via pip inside the env
RUN conda env create -f /workspace/environment.yml \
 && conda run -n leadlag python -m pip install --upgrade pip \
 && conda run -n leadlag python -m pip install -r /workspace/requirements.txt \
 && echo "conda activate leadlag" >> /root/.bashrc

ENV PATH /opt/conda/envs/leadlag/bin:$PATH

# Copy the rest of the source
COPY . /workspace

# Default command: show scenarios and how to run
CMD ["python", "hydra_main.py", "--scenario", "fixed_30", "--output_root", "results"]
