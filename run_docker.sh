#!/bin/bash
if [ -z "$1" ]
then 
	PORT=8888
else
	PORT=$1
fi

TAG=data
IMAGE=cspinc/fluvius
IMAGENAME=$IMAGE:$TAG 
IP=0.0.0.0
docker run --rm -it \
	-v $('pwd'):/content \
	-w /content \
	-p $PORT:$PORT \
	$IMAGENAME \
	/bin/bash
