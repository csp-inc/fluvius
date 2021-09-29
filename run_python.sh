#!/bin/bash
TAG=data
IMAGE=cspinc/fluvius
IMAGENAME=$IMAGE:$TAG
DPATH=docker/Dockerfile

if [[ "$(docker images -q $IMAGE 2> /dev/null)" == "" ]]; then
  echo "Image not found, building from recipe...."
  docker build --rm -t $IMAGENAME - < $DPATH
fi

# docker run --rm -td --ipc "host"  \
# 	-v $('pwd'):/content \
# 	-v /home/$USER/.vscode-server/:/root/.vscode-server/ \
# 	-w /content \
# 	$IMAGENAME
