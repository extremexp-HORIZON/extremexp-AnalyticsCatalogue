FROM python:3.9

# Install java for valentine
RUN apt update && apt install -y default-jre htop build-essential 
RUN apt-get install -y wget 

ENV NEO4J_HOST="neo4j://feature-discovery-neo4j:7687"
ENV NEO4J_USER="neo4j"
ENV NEO4J_PASSWORD="pass"
# ENV NEO4J_DATABASE="benchmark"

COPY requirements.txt /requirements.txt

# RUN 
# RUN wget  https://bootstrap.pypa.io/pip/3.8/get-pip.py  && python3 get-pip.py

RUN pip cache purge
RUN pip install setuptools==61

# RUN python3.8 -m
RUN pip install -r /requirements.txt
# RUN pip install fastapi uvicorn

COPY src /src
COPY pyproject.toml /pyproject.toml

# Install python deps
# RUN pip install -e .


CMD ["uvicorn", "src.feature_discovery.cli:app", "--host", "0.0.0.0", "--port", "8080"]
