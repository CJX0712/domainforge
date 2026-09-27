FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY domainforge ./domainforge
RUN pip install --no-cache-dir . && pip install --no-cache-dir pytest

CMD ["python", "-m", "domainforge.cli", "demo"]
