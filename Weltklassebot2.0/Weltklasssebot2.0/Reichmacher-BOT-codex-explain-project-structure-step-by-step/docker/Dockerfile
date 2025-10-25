FROM python:3.11-slim

ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=
ARG SWITCH_NO_ISOLATION=false

ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip setuptools wheel

RUN if [ "$SWITCH_NO_ISOLATION" = "true" ]; then \
      pip install -e ".[dev]" --no-build-isolation ; \
    else \
      pip install -e ".[dev]" ; \
    fi

ENV PYTHONHASHSEED=0

CMD ["bash", "-lc", "PYTHONPATH=src pytest -q"]
