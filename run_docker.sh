#!/usr/bin/env bash

PORT=${1:-8888}

TAG=data
IMAGE=cspinc/fluvius
IMAGENAME=$IMAGE:$TAG
IP=0.0.0.0

docker run --rm -it \
	-v $(pwd):/content \
	-w /content \
	-p $PORT:$PORT \
	$IMAGENAME \
	/bin/bash
