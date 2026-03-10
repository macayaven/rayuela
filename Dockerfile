# Project Rayuela — Analysis Environment
# Base: NVIDIA PyTorch image (ARM64-compatible, includes CUDA)
# NOTE: vLLM runs as a separate NVIDIA container (see docker-compose.yml)
FROM nvcr.io/nvidia/pytorch:26.02-py3

ARG RAYUELA_USER=carlos
ARG RAYUELA_GROUP=carlos
ARG RAYUELA_UID=1000
ARG RAYUELA_GID=1000

WORKDIR /workspace/rayuela

# Guard against PyTorch/CUDA replacement: extract torch-related packages
# from the base image as version-pinned constraints. This prevents pip
# from replacing NVIDIA's optimized PyTorch with a generic PyPI build
# when resolving dependencies for sentence-transformers or other packages.
#
# Uses `pip list --format=freeze` (not `pip freeze`) because the base
# image installs some packages from local .whl files — `pip freeze`
# records file paths that no longer exist, while `pip list --format=freeze`
# always outputs clean `package==version` entries.
RUN pip list --format=freeze | grep -iE '^(torch|nvidia|cuda|cudnn|nccl|triton)' \
    > /tmp/torch-constraints.txt \
    && echo "--- Protected packages ---" && cat /tmp/torch-constraints.txt

# Embedding models and NLP — constrained to protect torch
RUN pip install --no-cache-dir --constraint /tmp/torch-constraints.txt \
    sentence-transformers \
    transformers \
    tokenizers \
    accelerate \
    openai

# Embeddings & Dimensionality Reduction
RUN pip install --no-cache-dir \
    umap-learn \
    scikit-learn \
    openTSNE

# Topological Data Analysis (Phase 4+) — CPU-only, no torch dependency
# NOTE: giotto-tda deferred (no ARM64 wheel, needs source build investigation)
# ripser + persim cover core TDA needs; giotto-tda adds sklearn integration
RUN pip install --no-cache-dir \
    ripser \
    persim

# Visualization
RUN pip install --no-cache-dir \
    plotly \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    ipywidgets

# Graph analysis
RUN pip install --no-cache-dir \
    networkx \
    scipy

# Data handling
RUN pip install --no-cache-dir \
    pandas \
    tqdm

# NLP utilities for Spanish text
RUN pip install --no-cache-dir \
    spacy

# Download the large Spanish model (includes word vectors)
RUN python -m spacy download es_core_news_lg

RUN groupadd --gid "${RAYUELA_GID}" "${RAYUELA_GROUP}" \
    && useradd \
        --uid "${RAYUELA_UID}" \
        --gid "${RAYUELA_GID}" \
        --create-home \
        --shell /bin/bash \
        "${RAYUELA_USER}" \
    && mkdir -p "/home/${RAYUELA_USER}/.cache/huggingface" \
    && chown -R "${RAYUELA_UID}:${RAYUELA_GID}" /workspace/rayuela "/home/${RAYUELA_USER}"

ENV HOME=/home/${RAYUELA_USER}
ENV HF_HOME=${HOME}/.cache/huggingface
ENV PATH=${HOME}/.local/bin:${PATH}

USER ${RAYUELA_UID}:${RAYUELA_GID}

# JupyterLab only (vLLM runs in its own container on port 8000)
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
