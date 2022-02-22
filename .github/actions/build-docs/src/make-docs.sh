#!/usr/bin/env bash

ls -l .
cd docs
make html
cd ..

EXIT_STATUS=$?
echo "::set-output name=code::$EXIT_STATUS"
