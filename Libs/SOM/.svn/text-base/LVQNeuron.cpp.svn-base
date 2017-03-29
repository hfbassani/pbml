/* 
 * File:   LVQNeuron.cpp
 * Author: hans
 * 
 * Created on 28 de Fevereiro de 2012, 21:13
 */

#include "LVQNeuron.h"  

void LVQNeuron::setClass(int dClass) {
    this->dClass = dClass;
}

int LVQNeuron::getClass() {
    return this->dClass;
}

void LVQNeuron::setWon(bool won) {
    this->won = won;
}

bool LVQNeuron::getWon() {
    return this->won;
}

void LVQNeuron::setupWeights(uint inputSize) {
    DSNeuron::setupWeights(inputSize);

    avgDistanceN.size(inputSize);
    avgDistanceN.fill(0);
}

