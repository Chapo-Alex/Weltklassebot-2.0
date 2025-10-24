FROM python:3.11-slim
RUN pip install --upgrade pip
WORKDIR /app
COPY pyproject.toml ./
RUN pip install -e ".[dev]"
COPY . .
ENV PYTHONHASHSEED=0
CMD ["bash","-lc","PYTHONPATH=src pytest -q"]
