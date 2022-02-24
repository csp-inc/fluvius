#!/usr/bin/env bash

TAG=${1:-data}
IMAGE=cspinc/fluvius # TODO: get ghcr.io/csp-inc/fluvius working!
IMAGENAME=$IMAGE:$TAG
DPATH=docker/Dockerfile

if [[ "$(docker images -q $IMAGENAME 2> /dev/null)" == "" ]]; then
  echo "Image not found, building from recipe...."
  docker build --rm -t $IMAGENAME - < $DPATH
fi

docker run --rm -td --ipc "host" \
	-v $(pwd):/content \
	-v /home/$USER/.vscode-server/:/root/.vscode-server/ \
	-w /content \
	-p 8675:8000 \
	$IMAGENAME
