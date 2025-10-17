# Simple Dockerfile for reproducible runs
FROM condaforge/miniforge3:latest

WORKDIR /workspace

COPY environment.yml /workspace/environment.yml
RUN conda env create -f /workspace/environment.yml \
 && echo "conda activate leadlag" >> /root/.bashrc

ENV PATH /opt/conda/envs/leadlag/bin:$PATH

COPY . /workspace

# Default command: show scenarios and how to run
CMD ["python", "hydra_main.py", "--scenario", "fixed_30", "--output_root", "results"]

