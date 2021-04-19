#!/bin/bash

IMAGE=cspinc/fluvius
TAG=data
IMAGENAME=$IMAGE:$TAG
docker build --rm -t $IMAGENAME .
