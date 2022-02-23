#!/usr/bin/env bash

# debugging
echo "Evaluating: pwd"
pwd
echo "Evaluating: ls -a /src"
ls -a /src
echo "Evaluating: ls -a content"
ls -a content
echo "Evaluating: ls -a"
ls -a
echo "Evaluating: ls -a content/src"
ls -a content/src

mv content /content
cp /content/example_credentials /content/credentials # use dummy credentials to build docs
cd /content/docs
make html

EXIT_STATUS=$?
echo "::set-output name=code::$EXIT_STATUS"
