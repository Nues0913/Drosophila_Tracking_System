# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.13.0 AS uv
FROM ubuntu:26.04
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY python/ python/
COPY requirements/ requirements/
ARG ML_REQUIREMENTS=requirements/ml-cu126-py312.lock
RUN --mount=type=secret,id=proxy_ca,required=false \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    uv python install 3.12 && uv venv --python 3.12 /opt/flytrack-ml && \
    uv pip install --python /opt/flytrack-ml/bin/python --require-hashes -r "${ML_REQUIREMENTS}" && \
    uv pip install --python /opt/flytrack-ml/bin/python --no-deps -e .
ENV VIRTUAL_ENV=/opt/flytrack-ml PATH="/opt/flytrack-ml/bin:${PATH}" PYTHONUNBUFFERED=1
COPY configs/ configs/
COPY scripts/ scripts/
CMD ["python", "-m", "flytrack_train", "ppo", "--config", "configs/train/ppo.yaml"]
