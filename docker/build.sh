#!/usr/bin/env bash

TAG=${1:-data}
IMAGE=cspinc/fluvius

IMAGENAME=$IMAGE:$TAG

docker build --rm -t $IMAGENAME .
