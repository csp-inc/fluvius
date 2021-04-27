#!/bin/bash
TAG=data
IMAGE=cspinc/fluvius
IMAGENAME=$IMAGE:$TAG 
PORT=8080
IP=0.0.0.0
#blobcontainer=$BLOBCONTAINER
docker run --rm \
	-v $('pwd'):/content\
	-w /content\
	-p $PORT:$PORT\
	$IMAGENAME\
	jupyter notebook --port $PORT --ip $IP --no-browser --allow-root
