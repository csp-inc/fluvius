#!/usr/bin/env bash

TAG=${1:-gpu}
IMAGE=cspinc/fluvius
IMAGENAME=$IMAGE:$TAG
DPATH=docker/gpu.Dockerfile

if [[ "$(docker images -q $IMAGENAME 2> /dev/null)" == "" ]]; then
  echo "Image not found, building from recipe...."
  docker build --rm -t $IMAGENAME - < $DPATH
fi

docker run --rm -td --ipc "host"  \
	--gpus all \
	-v $('pwd'):/content \
	-v /home/$USER/.vscode-server/:/root/.vscode-server/ \
	-w /content \
	$IMAGENAME
