#!/bin/sh

BIN_NAME="./wip"
PARAM="../../Parameters/std25_0"
TEST_FILE="../../Parameters/inputPathsTestVowel"
TRAIN_FILE="../../Parameters/inputPathsTrainVowel"
RESULT_FOLDER="../wip-std25-70n"

# tmux new-session -d "${BIN_NAME} -i ${TRAIN_FILE}01 -t ${TEST_FILE} -r ${RESULT_FOLDER}-l01/ -p ${PARAM} -s -c"
# tmux new-session -d "${BIN_NAME} -i ${TRAIN_FILE}05 -t ${TEST_FILE} -r ${RESULT_FOLDER}-l05/ -p ${PARAM} -s -c"
# tmux new-session -d "${BIN_NAME} -i ${TRAIN_FILE}10 -t ${TEST_FILE} -r ${RESULT_FOLDER}-l10/ -p ${PARAM} -s -c"
# tmux new-session -d "${BIN_NAME} -i ${TRAIN_FILE}25 -t ${TEST_FILE} -r ${RESULT_FOLDER}-l25/ -p ${PARAM} -s -c"
# tmux new-session -d "${BIN_NAME} -i ${TRAIN_FILE}50 -t ${TEST_FILE} -r ${RESULT_FOLDER}-l50/ -p ${PARAM} -s -c"
# tmux new-session -d "${BIN_NAME} -i ${TRAIN_FILE}75 -t ${TEST_FILE} -r ${RESULT_FOLDER}-l75/ -p ${PARAM} -s -c"
tmux new-session -d "${BIN_NAME} -i ${TRAIN_FILE} -t ${TEST_FILE} -r ${RESULT_FOLDER}-l100/ -p ${PARAM} -s -c -N 70"