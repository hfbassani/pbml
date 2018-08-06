#!/usr/bin/env bash

convert -density 200 "$1"/*.pdf -set filename:base "%[base]" "%[filename:base].png"