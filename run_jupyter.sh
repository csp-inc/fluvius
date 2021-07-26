#!/bin/bash
if [ -z "$1" ]
then 
	PORT=8888
else
	PORT=$1
fi

TAG=latest
IMAGE=cspinc/fluvius
IMAGENAME=$IMAGE:$TAG 
IP=0.0.0.0
#BLOB=$BLOBCONTAINER
docker run --rm \
	-v $('pwd'):/content \
	-w /content \
	-p $PORT:$PORT \
	$IMAGENAME \
	jupyter notebook --port $PORT --ip $IP --no-browser --allow-root
