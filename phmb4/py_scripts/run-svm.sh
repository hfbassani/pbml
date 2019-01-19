#!/bin/sh

PARAM="../../Parameters/SVM_0"
TRAIN_FOLDER="../../Datasets/Realdata_3Times3FoldsExp_Train"
RESULT_FOLDER="svm-msc/"

# tmux new-session -d "python svm.py  -i ${TRAIN_FOLDER}S01 -p ${PARAM} -o ${RESULT_FOLDER} -s 0.01"
tmux new-session -d "python svm.py  -i ${TRAIN_FOLDER}S05 -p ${PARAM} -o ${RESULT_FOLDER} -s 0.05"
tmux new-session -d "python svm.py  -i ${TRAIN_FOLDER}S10 -p ${PARAM} -o ${RESULT_FOLDER} -s 0.10"
tmux new-session -d "python svm.py  -i ${TRAIN_FOLDER}S25 -p ${PARAM} -o ${RESULT_FOLDER} -s 0.25"
tmux new-session -d "python svm.py  -i ${TRAIN_FOLDER}S50 -p ${PARAM} -o ${RESULT_FOLDER} -s 0.50"
tmux new-session -d "python svm.py  -i ${TRAIN_FOLDER}S75 -p ${PARAM} -o ${RESULT_FOLDER} -s 0.75"
tmux new-session -d "python svm.py  -i ${TRAIN_FOLDER} -p ${PARAM} -o ${RESULT_FOLDER} -s 1.0"
