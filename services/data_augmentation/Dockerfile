FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y apt-utils python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


RUN apt-get clean && rm -rf /var/lib/apt/lists/*



RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src


# Volume
RUN mkdir -p /app/resources/

# Expose ports for FastAPI and MLflow
EXPOSE 8000 5005

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
