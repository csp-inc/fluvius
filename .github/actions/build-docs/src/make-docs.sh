#!/usr/bin/env bash

cd docs
make html
cd ..

EXIT_STATUS=$?
echo "::set-output name=code::$EXIT_STATUS"
