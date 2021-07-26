#!/bin/bash
TAG=data
IMAGE=cspinc/fluvius
IMAGENAME=$IMAGE:$TAG
DPATH=docker/Dockerfile

if [[ "$(docker images -q $IMAGE 2> /dev/null)" == "" ]]; then
  echo "Image not found, building from recipe...."
  docker build --rm -t $IMAGENAME - < $DPATH
fi

docker run --rm -td \
	-v $('pwd'):/fluvius \
	-v /home/.vscode-server/extensions:/root/.vscode-server/extensions \
	-w /fluvius \
	$IMAGENAME 
