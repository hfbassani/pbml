#!/bin/sh

PARAM_NAME="../Parameters/parametersNameSSSOM-WIP"
TEST_FOLDER="../../Datasets/Realdata_3Times3FoldsExp_Test2"
PARAM="wip2_0"
FOLDER="wip2-120n"

python accuracy_calc.py -t ${TEST_FOLDER} -i ../Models/${FOLDER}-l01 -r 500 -o ${FOLDER}/${FOLDER}-l01 -p ../Parameters/${PARAM} -n ${PARAM_NAME}
python accuracy_calc.py -t ${TEST_FOLDER} -i ../Models/${FOLDER}-l05 -r 500 -o ${FOLDER}/${FOLDER}-l05 -p ../Parameters/${PARAM} -n ${PARAM_NAME}
python accuracy_calc.py -t ${TEST_FOLDER} -i ../Models/${FOLDER}-l10 -r 500 -o ${FOLDER}/${FOLDER}-l10 -p ../Parameters/${PARAM} -n ${PARAM_NAME}
python accuracy_calc.py -t ${TEST_FOLDER} -i ../Models/${FOLDER}-l25 -r 500 -o ${FOLDER}/${FOLDER}-l25 -p ../Parameters/${PARAM} -n ${PARAM_NAME}
python accuracy_calc.py -t ${TEST_FOLDER} -i ../Models/${FOLDER}-l50 -r 500 -o ${FOLDER}/${FOLDER}-l50 -p ../Parameters/${PARAM} -n ${PARAM_NAME}
python accuracy_calc.py -t ${TEST_FOLDER} -i ../Models/${FOLDER}-l75 -r 500 -o ${FOLDER}/${FOLDER}-l75 -p ../Parameters/${PARAM} -n ${PARAM_NAME}
python accuracy_calc.py -t ${TEST_FOLDER} -i ../Models/${FOLDER}-l100 -r 500 -o ${FOLDER}/${FOLDER}-l100 -p ../Parameters/${PARAM} -n ${PARAM_NAME}