#!/bin/sh

BIN_NAME="./wip"
PARAM="../../Parameters/wip-cluster17_0"
TRAIN_FILE="../../Parameters/inputPathsTrain00"
RESULT_FOLDER="../wip-cluster17"

tmux new-session -d "${BIN_NAME} -i ${TRAIN_FILE} -r ${RESULT_FOLDER}-l00/ -p ${PARAM} -s -c -n"