FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 SHADOWTRACK_ROOT=/workspace
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 libgomp1 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY licenses ./licenses
RUN pip install -e ".[web,detection-cpu,sync]"
EXPOSE 8000
CMD ["shadowtrack","mvp","serve","--host","0.0.0.0","--port","8000"]
