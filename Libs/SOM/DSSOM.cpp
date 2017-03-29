/* 
 * File:   FSOM.cpp
 * Author: hans
 * 
 * Created on 6 de Outubro de 2010, 17:41
 */
#ifndef DSSOM_CPP
#define DSSOM_CPP

#include <cmath>

#include "DSSOM.h"
#include "DebugOut.h"
#include <math.h>

template <class TNeuron, class TSOMParameters>
DSSOM<TNeuron, TSOMParameters>::DSSOM() {
    this->section = "DSSOM";
    this->comments = "Dimension Selective Self-Organizing Map";
    this->callback = NULL;
}

template <class TNeuron, class TSOMParameters>
void DSSOM<TNeuron, TSOMParameters>::setParameters(uint numNeighbors, float alpha, float beta, float lambda2, float sigma, float lambda1, uint tmax) {
    this->parameters.numNeighbors = numNeighbors;
    this->parameters.alpha = alpha;
    this->parameters.lambda2 = lambda2;
    this->parameters.sigma = sigma;
    this->parameters.lambda1 = lambda1;
    this->parameters.tmax = tmax;
    this->parameters.beta = beta;
}

template <class TNeuron, class TSOMParameters>
void DSSOM<TNeuron, TSOMParameters>::initialize(uint somRows, uint somCols, uint inputSize) {
    this->somRows = somRows;
    this->somCols = somCols;
    this->inputSize = inputSize;
    resetTime();

    this->neurons.size(somRows, somCols);
    this->activationMap.size(somRows, somCols);

    reRandomizeWeights();
}

template <class TNeuron, class TSOMParameters>
void DSSOM<TNeuron, TSOMParameters>::reRandomizeWeights() {
    //srand(time(NULL));

    for (uint r = 0; r < this->neurons.rows(); r++)
        for (uint c = 0; c < this->neurons.cols(); c++) {
            this->neurons[r][c].set(r, c);
            this->neurons[r][c].setupWeights(this->inputSize);
        }

    this->activationMap.fill(0);
}

template <class TNeuron, class TSOMParameters>
float DSSOM<TNeuron, TSOMParameters>::presentPaternToNeuron(const MatVector<float> &data, TNeuron &neuron) {
    float distance = 0;

    for (uint i = 0; i < data.size(); i++) {
        distance += neuron.dsWeights[i] * qrt(fabs(data[i] - neuron.weights[i]));
    }

    //    neuron.dsWeights.mqrt();
    //    float sum = neuron.dsWeights.sum();
    //    distance = distance/(1+sqrt(sum));

    //    distance = distance/(1+neuron.dsWeights.sum());
    //    return 1/(distance+1);

    return (neuron.dsWeights.sum() / (neuron.dsWeights.sum() + distance + 0.0000001));
}

template <class TNeuron, class TSOMParameters>
void DSSOM<TNeuron, TSOMParameters>::updateNeuron(TNeuron &neuron, MatVector<float> &data, float alpha, float phi) {


    for (uint i = 0; i < data.size(); i++) {
        //update neuron weights
        float distance = fabs(data[i] - neuron.weights[i]);
        neuron.avgDistance[i] = phi * beta_t * distance + (1 - phi * beta_t) * neuron.avgDistance[i];
    }

    float max = neuron.avgDistance.max();

    //update neuron fs weights
    for (uint i = 0; i < data.size(); i++) {
        if (max != 0)
            neuron.dsWeights[i] = 1 - (neuron.avgDistance[i] / max);
        else
            neuron.dsWeights[i] = 1;

        if (neuron.dsWeights[i] < this->parameters.minDW)
            neuron.dsWeights[i] = this->parameters.minDW;
    }

    //update neuron weights
    for (uint i = 0; i < data.size(); i++) {

        neuron.weights[i] = neuron.weights[i] + phi * alpha * (data[i] - neuron.weights[i]);
    }
}

template <class TNeuron, class TSOMParameters>
void DSSOM<TNeuron, TSOMParameters>::resetTime() {
    this->t = 0;
    //alpha_t = ((parameters.tmax-t)*parameters.alpha)/parameters.tmax;  //linear decay
    this->alpha_t = this->parameters.alpha * exp(-(float) this->t / (this->parameters.tmax * this->parameters.lambda2)); //exponencial decay
    this->beta_t = this->parameters.beta;
    this->sigma_t = this->parameters.sigma * exp(-((int) this->t + 1) / (this->parameters.tmax * this->parameters.lambda1));
}

template <class TNeuron, class TSOMParameters>
bool DSSOM<TNeuron, TSOMParameters>::incTime() {
    this->t++;
    if (this->t < this->parameters.tmax) {
        //alpha_t = ((parameters.tmax-t)*parameters.alpha)/parameters.tmax;  //linear decay
        this->alpha_t = this->parameters.alpha * exp(-(float) this->t / (this->parameters.tmax * this->parameters.lambda2)); //exponencial decay
        if (this->alpha_t < 0.00001) this->alpha_t = 0.00001;
        this->sigma_t = this->parameters.sigma * exp(-((int) this->t + 1) / (this->parameters.tmax * this->parameters.lambda1));
        if (this->sigma_t < 0.000001) this->sigma_t = 0.000001;
    }

    //dbgOut(0) <<" alpha:" << this->alpha_t << " sigma:" << this->sigma_t << " t: " << this->t << endl;
    return this->t < this->parameters.tmax;
}

template <class TNeuron, class TSOMParameters>
bool DSSOM<TNeuron, TSOMParameters>::isNoise(const MatVector<float> &data) {
    TNeuron bmu;
    this->findBMU(data, bmu);
    return (bmu.activation<this->parameters.outliersThreshold);
}

#endif //DSSOM_CPP
