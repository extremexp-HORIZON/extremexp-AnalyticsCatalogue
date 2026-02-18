#!/bin/bash

# ingest the data into the neo4j database
feature-discovery-cli ingest-data --data-discovery-threshold=0.7 --discover-connections-data-lake

# run autofeat for dataset
feature-discovery-cli run-autofeat --dataset-labels crypto_desktop