#!/usr/bin/env bash

WD=$(pwd)
mv content /content
cp /content/example_credentials /content/credentials # use dummy credentials to build docs
cd /content/docs
make html

echo "Evaluating: ls -a build/html"
ls -a build/html

cd $WD
mv /content content

EXIT_STATUS=$?
echo "::set-output name=code::$EXIT_STATUS"
