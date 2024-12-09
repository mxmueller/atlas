# services/mask-generation/Dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV FORCE_CUDA=1

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install "numpy>=1.23.5,<2.0.0"
RUN pip3 install --no-cache-dir torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]