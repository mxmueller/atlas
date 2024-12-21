#!/bin/bash

# Name des Docker-Images
IMAGE_NAME="randomizer-testcase-generator"

# Docker-Image bauen
echo "Build..."
docker build -t $IMAGE_NAME .

# Container starten
docker run -d -p 5000:5000 --name $IMAGE_NAME"_container" $IMAGE_NAME

