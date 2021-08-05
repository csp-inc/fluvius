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
DPATH=docker/Dockerfile

if [[ "$(docker images -q $IMAGE 2> /dev/null)" == "" ]]; then
  echo "Image not found, building from recipe...."
  docker build --rm -t $IMAGENAME - < $DPATH
fi
 
IP=0.0.0.0
#BLOB=$BLOBCONTAINER
docker run --rm \
	-v $('pwd'):/content \
	-w /content \
	-p $PORT:$PORT \
	$IMAGENAME \
	jupyter notebook --port $PORT --ip $IP --no-browser --allow-root
