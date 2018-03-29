#!/bin/bash

for FILE in "$1"/*.pdf; do
  pdfcrop "${FILE}"
done
