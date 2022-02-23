#!/usr/bin/env bash

pwd
ls -la .
cd /content
ls -la .
ls -la docs
ls -la src
cd docs
make html
cd ..

EXIT_STATUS=$?
echo "::set-output name=code::$EXIT_STATUS"
