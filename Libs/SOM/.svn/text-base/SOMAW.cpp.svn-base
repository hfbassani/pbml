/* 
 * File:   FSOM.cpp
 * Author: hans
 * 
 * Created on 6 de Outubro de 2010, 17:41
 */
#ifndef SOMAW_CPP
#define SOMAW_CPP

#include <cmath>

#include "SOMAW.h"
#include "DebugOut.h"
#include <math.h>

template <class TNeuron, class TSOMParameters>
SOMAW<TNeuron, TSOMParameters>::SOMAW() {
    this->section = "SOMAD";
    this->comments = "Adaptive Weighting SOM (Kangas 1990)";
    this->callback = NULL;
}

template <class TNeuron, class TSOMParameters>
void SOMAW<TNeuron, TSOMParameters>::setParameters(uint numNeighbors, float alpha, float lambda2, float sigma, float lambda1, uint tmax, float k1, float k2, float k3) {
    this->parameters.numNeighbors = numNeighbors;
    this->parameters.alpha = alpha;
    this->parameters.lambda2 = lambda2;
    this->parameters.sigma = sigma;
    this->parameters.lambda1 = lambda1;
    this->parameters.tmax = tmax;
    this->parameters.k1 = k1;
    this->parameters.k2 = k2;
    this->parameters.k3 = k3;
}

template <class TNeuron, class TSOMParameters>
void SOMAW<TNeuron, TSOMParameters>::initialize(uint somRows, uint somCols, uint inputSize) {
    this->somRows = somRows;
    this->somCols = somCols;
    this->inputSize = inputSize;
    resetTime();

    this->neurons.size(somRows, somCols);
    this->activationMap.size(somRows, somCols);

    reRandomizeWeights();
}

template <class TNeuron, class TSOMParameters>
void SOMAW<TNeuron, TSOMParameters>::reRandomizeWeights() {
    //srand(time(NULL));

    for (uint r = 0; r < this->neurons.rows(); r++)
        for (uint c = 0; c < this->neurons.cols(); c++) {
            this->neurons[r][c].set(r, c);
            this->neurons[r][c].setupWeights(this->inputSize);
        }

    this->activationMap.fill(0);
}

template <class TNeuron, class TSOMParameters>
float SOMAW<TNeuron, TSOMParameters>::presentPaternToNeuron(const MatVector<float> &data, TNeuron &neuron) {
    float distance = 0;

        for (uint i = 0; i < data.size(); i++) {
            distance += qrt(neuron.dsWeights[i]) * qrt(fabs(data[i] - neuron.weights[i]));
        }

    return (1 / (distance + 0.0000001));
}

template <class TNeuron, class TSOMParameters>
void SOMAW<TNeuron, TSOMParameters>::updateNeuron(TNeuron &neuron, MatVector<float> &data, float alpha, float phi) {


    for (uint i = 0; i < data.size(); i++) {
        //update neuron weights
        float distance = fabs(data[i] - neuron.weights[i]);
            neuron.avgDistance[i] = neuron.avgDistance[i] + this->parameters.k1 * ((neuron.dsWeights[i] * distance) - neuron.avgDistance[i]);
        }

    float average = neuron.avgDistance.sum() / neuron.avgDistance.size();

    //update neuron ds weights
    for (uint i = 0; i < data.size(); i++) {
        //update neuron weights
        float distance = fabs(data[i] - neuron.weights[i]);

            if (neuron.dsWeights[i] * distance > average)
                neuron.dsWeights[i] = this->parameters.k2 * neuron.dsWeights[i];
            else
                neuron.dsWeights[i] = this->parameters.k3 * neuron.dsWeights[i];
        }

    float sum  = neuron.dsWeights.sum();
    for (uint i = 0; i < data.size(); i++) {
            neuron.dsWeights[i] = neuron.dsWeights[i] / sum;
        }

    //update neuron weights
    for (uint i = 0; i < data.size(); i++) {
            neuron.weights[i] = neuron.weights[i] + phi * alpha * (data[i] - neuron.weights[i]);
        }
}

template <class TNeuron, class TSOMParameters>
void SOMAW<TNeuron, TSOMParameters>::resetTime() {
    this->t = 0;
    //alpha_t = ((parameters.tmax-t)*parameters.alpha)/parameters.tmax;  //linear decay
    this->alpha_t = this->parameters.alpha * exp(-(float) this->t / (this->parameters.tmax * this->parameters.lambda2)); //exponencial decay
    this->sigma_t = this->parameters.sigma * exp(-((int) this->t + 1) / (this->parameters.tmax * this->parameters.lambda1));
}

template <class TNeuron, class TSOMParameters>
bool SOMAW<TNeuron, TSOMParameters>::incTime() {
    this->t++;
    if (this->t < this->parameters.tmax) {
        //alpha_t = ((parameters.tmax-t)*parameters.alpha)/parameters.tmax;  //linear decay
        this->alpha_t = this->parameters.alpha * exp(-(float) this->t / (this->parameters.tmax * this->parameters.lambda2)); //exponencial decay
        if (this->alpha_t < 0.00001) this->alpha_t = 0.00001;
        this->sigma_t = this->parameters.sigma * exp(-((int) this->t + 1) / (this->parameters.tmax * this->parameters.lambda1));
    }

    return this->t < this->parameters.tmax;
}

#endif //SOMAW_CPP
