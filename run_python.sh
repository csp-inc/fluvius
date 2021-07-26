#!/bin/bash
TAG=data
IMAGE=cspinc/fluvius
IMAGENAME=$IMAGE:$TAG
DPATH=docker/Dockerfile

if [[ "$(docker images -q $IMAGE 2> /dev/null)" == "" ]]; then
  echo "Image not found, building from recipe...."
  docker build --rm -t $IMAGENAME - < $DPATH
fi

BLOB=$BLOBCONTAINER
docker run --rm \
	-v $('pwd'):/fluvius \
	-w /fluvius \
	-v $BLOB:/blob \
	$IMAGENAME 
