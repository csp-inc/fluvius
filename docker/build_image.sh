#!/bin/bash
if [ -z "$1" ] 
then 
	TAG=data
else
	TAG=$1
fi
IMAGE=cspinc/fluvius
IMAGENAME=$IMAGE:$TAG
docker build --rm -t $IMAGENAME .
