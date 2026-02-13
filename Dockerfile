
FROM python:3.9
RUN apt-get update && apt-get upgrade -y && apt-get install apt-utils
RUN mkdir /app/
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY portal /app/
COPY docker-compose.yaml docker-compose.yaml
COPY config.ini config.ini
EXPOSE 9000
CMD ["streamlit", "run", "main.py", "--server.port=9000", "--server.address=0.0.0.0"]