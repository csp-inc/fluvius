#!/usr/bin/env bash

pwd
ls -la .
ls /content
cd /content/docs
ls -la /content/docs/src
make html
cd ..

EXIT_STATUS=$?
echo "::set-output name=code::$EXIT_STATUS"
