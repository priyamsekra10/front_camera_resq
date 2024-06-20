FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    cmake \
    && apt-get clean
    
RUN pip install -r requirements.txt

# FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

EXPOSE 9092

CMD ["python", "kafka_1.py"]
