#!/usr/bin/env bash

for file in "$1"/*-crop.pdf;
do
    mv "$file" "${file/-crop/}"
done