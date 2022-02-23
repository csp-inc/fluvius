#!/usr/bin/env bash

echo "pwd"
pwd
echo "ls -a /src"
ls -a /src
echo "ls -a /content"
ls -a /content
echo "ls -a"
ls -a
cd /content/docs
echo "ls -a /content/src"
ls -a /content/src
make html

EXIT_STATUS=$?
echo "::set-output name=code::$EXIT_STATUS"
