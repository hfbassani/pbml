#!/bin/sh

METRICS="CE"
PARAM_NAME="../Parameters/parametersNameSSSOM-WIP"
TEST_FOLDER="../../Datasets/Realdata"
PARAM="../Parameters/wip-final_0"
FOLDER="wip-final-l00"

java -jar ClusteringAnalysis.jar ${METRICS} ${TEST_FOLDER} ../Models/${FOLDER} ${FOLDER} -p ${PARAM} -n ${PARAM_NAME} -r 500 -t -S