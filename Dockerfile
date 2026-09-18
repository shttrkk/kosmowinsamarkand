# CPU-образ инференса (работает на любой машине с Docker, GPU не нужен).
#
#   docker build -t firemon .
#   docker run --rm -v /path/to/test:/data:ro -v /path/to/out:/out firemon --data-dir /data --output /out/submission.csv
#
# Внутри: код, конфиги, веса из weights/ и все зависимости (включая torch CPU — для нейросетевого BS-бэкенда).
# Сеть при запуске не нужна. GPU-вариант — Dockerfile.gpu.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 — OpenMP для LightGBM; libexpat1 — нужен колесу rasterio (в slim-образе его нет); GDAL/PROJ/GEOS
# встроены в колёса rasterio/pyproj/shapely
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# зависимости — отдельным слоем: пересборка после правки кода занимает секунды
COPY requirements-base.txt requirements-cpu.txt ./
RUN pip install -r requirements-cpu.txt

COPY pyproject.toml inference.py train.py ./
COPY configs ./configs
COPY src ./src
COPY weights ./weights

ENTRYPOINT ["python", "inference.py"]
CMD ["--help"]
