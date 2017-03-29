/* 
 * File:   FSOM.cpp
 * Author: hans
 * 
 * Created on 6 de Outubro de 2010, 17:41
 */
#ifndef DSSOMC_CPP
#define DSSOMC_CPP

#include <cmath>

#include "DSSOMC.h"
#include "DebugOut.h"
#include <math.h>

template <class TNeuron, class TSOMParameters>
DSSOMC<TNeuron, TSOMParameters>::DSSOMC() {
    this->section = "DSSOM";
    this->comments = "Dimension Selective Self-Organizing Map";
    this->callback = NULL;
}

template <class TNeuron, class TSOMParameters>
void DSSOMC<TNeuron, TSOMParameters>::setParameters(uint numNeighbors, float alpha, float beta, float lambda2, float sigma, float lambda1, uint tmax) {
    this->parameters.numNeighbors = numNeighbors;
    this->parameters.alpha = alpha;
    this->parameters.lambda2 = lambda2;
    this->parameters.sigma = sigma;
    this->parameters.lambda1 = lambda1;
    this->parameters.tmax = tmax;
    this->parameters.beta = beta;
}

template <class TNeuron, class TSOMParameters>
void DSSOMC<TNeuron, TSOMParameters>::initialize(uint somRows, uint somCols, uint inputSize) {
    this->somRows = somRows;
    this->somCols = somCols;
    this->inputSize = inputSize;
    resetTime();

    this->neurons.size(somRows, somCols);
    this->activationMap.size(somRows, somCols);
    fixed.size(inputSize);
    fixed.fill(-1);

    reRandomizeWeights();
}

template <class TNeuron, class TSOMParameters>
void DSSOMC<TNeuron, TSOMParameters>::initClasses() {
    int half = this->neurons.rows()*this->neurons.cols()/2;
    
    for (uint r = 0; r < this->neurons.rows(); r++)
    for (uint c = 0; c < this->neurons.cols(); c++) {
        if ((r+1)*(c+1)<half)
            this->neurons[r][c].weights[this->inputSize-1] = 0;
        else
            this->neurons[r][c].weights[this->inputSize-1] = 1;
    }
}

template <class TNeuron, class TSOMParameters>
void DSSOMC<TNeuron, TSOMParameters>::reRandomizeWeights() {
    //srand(time(NULL));

    for (uint r = 0; r < this->neurons.rows(); r++)
        for (uint c = 0; c < this->neurons.cols(); c++) {
            this->neurons[r][c].set(r, c);
            this->neurons[r][c].setupWeights(this->inputSize);
        }

    this->activationMap.fill(0);
}

template <class TNeuron, class TSOMParameters>
float DSSOMC<TNeuron, TSOMParameters>::presentPaternToNeuron(const MatVector<float> &data, TNeuron &neuron) {
    float distance = 0;

    for (uint i = 0; i < data.size(); i++) {
        distance += neuron.dsWeights[i] * qrt(fabs(data[i] - neuron.weights[i]));
    }

    //    neuron.dsWeights.mqrt();
    //    float sum = neuron.dsWeights.sum();
    //    distance = distance/(1+sqrt(sum));

    //    distance = distance/(1+neuron.dsWeights.sum());
    return 1/(1+distance);

    //return (neuron.dsWeights.sum() / (neuron.dsWeights.sum() + distance + 0.0000001));
}

template <class TNeuron, class TSOMParameters>
void DSSOMC<TNeuron, TSOMParameters>::findBMU_NC(int classW, TNeuron &bmuNC) {
    
    float actvationBest = 0;
    int wr=-1, wc=-1;
    int mdr, mdc;
    float hdif = 0;
    for (int r=0; r<this->activationMap.rows(); r++)
    for (int c=0; c<this->activationMap.cols(); c++) {
        float dif = fabs(classW - this->neurons[r][c].weights[this->inputSize-1]);
        if (dif>0.5){
            if (this->activationMap[r][c]>actvationBest) {
                actvationBest = this->activationMap[r][c];
                wr=r;
                wc=c;
            }
        } else {
            if (dif>hdif) {
                hdif = dif;
                mdr = r;
                mdc = c;
            }
        }
    }
    
    if (wr<0)
        bmuNC = this->neurons[mdr][mdc];
    else
        bmuNC = this->neurons[wr][wc];
}

template <class TNeuron, class TSOMParameters>
void DSSOMC<TNeuron, TSOMParameters>::updateNeuron(TNeuron &neuron, MatVector<float> &data, float alpha, float phi) {

//    if (fabs(data[data.size()-1] - neuron.weights[data.size()-1]) <0.5) {
        //Data is from the same class
    for (uint i = 0; i < data.size(); i++) {
        //update neuron weights
        float distance = fabs(data[i] - neuron.weights[i]);
        neuron.avgDistance[i] = phi * beta_t * distance + (1 - phi * beta_t) * neuron.avgDistance[i];
    }
//    } else {
//        //Data is from the a different class
//        for (uint i = 0; i < data.size(); i++) {
//            //update neuron weights
//            float distance = fabs(data[i] - neuron.weights[i]);
//            neuron.avgDistanceN[i] = phi * beta_t * distance + (1 - phi * beta_t) * neuron.avgDistanceN[i];
//        }
//    }

    TNeuron bmuNC;
    findBMU_NC(neuron.weights[this->inputSize-1], bmuNC);
    if (bmuNC.c!=neuron.c)
    for (uint i = 0; i < data.size(); i++) {
        //update neuron weights
        float distance = 1-fabs(bmuNC.weights[i] - neuron.weights[i]);
        neuron.avgDistance[i] = phi * beta_t * distance + (1 - phi * beta_t) * neuron.avgDistance[i];
    }  
    
    neuron.avgDistance[data.size()-1] = 0;
   
    float max =  neuron.avgDistance.max();
    
    //update neuron fs weights
    for (uint i = 0; i < data.size(); i++) {
        
        if (fixed[i]>=0) {
            neuron.dsWeights[i] = fixed[i];
            continue;
        }
        
        if (max != 0)
            neuron.dsWeights[i] = 1-(neuron.avgDistance[i] / max);
        else
            neuron.dsWeights[i] = 1;
        
        if (neuron.dsWeights[i] < this->parameters.minDW)
            neuron.dsWeights[i] = this->parameters.minDW;
    }

    //update neuron weights
    if (fabs(data[this->inputSize-1]-neuron.weights[this->inputSize-1])<0.5) {
        for (uint i = 0; i < data.size()-1; i++) {
            neuron.weights[i] = neuron.weights[i] + phi * alpha * (data[i] - neuron.weights[i]);
        }
    } else {
        for (uint i = 0; i < data.size()-1; i++) {
            neuron.weights[i] = neuron.weights[i] - phi * (alpha/10) * (data[i] - neuron.weights[i]);
        }
    }
}

template <class TNeuron, class TSOMParameters>
void DSSOMC<TNeuron, TSOMParameters>::resetTime() {
    this->t = 0;
    //alpha_t = ((parameters.tmax-t)*parameters.alpha)/parameters.tmax;  //linear decay
    this->alpha_t = this->parameters.alpha * exp(-(float) this->t / (this->parameters.tmax * this->parameters.lambda2)); //exponencial decay
    this->beta_t = this->parameters.beta;
    this->sigma_t = this->parameters.sigma * exp(-((int) this->t + 1) / (this->parameters.tmax * this->parameters.lambda1));
}

template <class TNeuron, class TSOMParameters>
bool DSSOMC<TNeuron, TSOMParameters>::incTime() {
    this->t++;
    if (this->t < this->parameters.tmax) {
        //alpha_t = ((parameters.tmax-t)*parameters.alpha)/parameters.tmax;  //linear decay
        this->alpha_t = this->parameters.alpha * exp(-(float) this->t / (this->parameters.tmax * this->parameters.lambda2)); //exponencial decay
        if (this->alpha_t < 0.00001) this->alpha_t = 0.00001;
        this->sigma_t = this->parameters.sigma * exp(-((int) this->t + 1) / (this->parameters.tmax * this->parameters.lambda1));
    }

    return this->t < this->parameters.tmax;
}

template <class TNeuron, class TSOMParameters>
void DSSOMC<TNeuron, TSOMParameters>::getMaxRelevances(MatVector<float> &maxRelevances) {
    maxRelevances.size(this->inputSize);
    maxRelevances.fill(0);

    for (uint r = 0; r < this->neurons.rows(); r++)
        for (uint c = 0; c < this->neurons.cols(); c++) {

            //Compute max relevance for each dimension
            for (uint i = 0; i < maxRelevances.size(); i++) {
                    if (maxRelevances[i] < this->neurons[r][c].dsWeights[i])
                        maxRelevances[i] = this->neurons[r][c].dsWeights[i];
                }

        }
}

template <class TNeuron, class TSOMParameters>
void DSSOMC<TNeuron, TSOMParameters>::getRelevances(MatVector<float> &relevances) {

    relevances.size(this->inputSize);
    relevances.fill(0);

    MatVector<float> average(relevances.size());
    average.fill(0);

    for (uint r = 0; r < this->neurons.rows(); r++)
        for (uint c = 0; c < this->neurons.cols(); c++) {

            //Compute max relevance for each dimension
            for (uint i = 0; i < relevances.size(); i++) {
                if (relevances[i] < this->neurons[r][c].dsWeights[i])
                    relevances[i] = this->neurons[r][c].dsWeights[i];
            }

            //Compute average neurons positions
            average.add(this->neurons[r][c].weights);
        }

    average.mult(1 / (float) this->neurons.size());

    MatVector<float> variance(relevances.size());
    variance.fill(0);

    //Compute neuron positions variance
    for (uint r = 0; r < this->neurons.rows(); r++)
        for (uint c = 0; c < this->neurons.cols(); c++) {
            MatVector<float> dif = average;
            dif.mult(-1);
            dif.add(this->neurons[r][c].weights);
            dif.vabs();
            dif.vqrt();
            variance.add(dif);
        }

    //Compute relevances
    for (uint i = 0; i < this->inputSize; i++) {
        relevances[i] = relevances[i] * variance[i];
    }

    relevances.mult(1 / relevances.max());
}

template <class TNeuron, class TSOMParameters>
void DSSOMC<TNeuron, TSOMParameters>::getVariances(MatVector<float> &variances) {

    MatVector<float> average(this->inputSize);
    average.fill(0);

    for (uint r = 0; r < this->neurons.rows(); r++)
        for (uint c = 0; c < this->neurons.cols(); c++) {
            //Compute average neurons positions
            average.add(this->neurons[r][c].weights);
        }

    average.mult(1 / (float) this->neurons.size());


    variances.size(this->inputSize);
    variances.fill(0);

    //Compute neuron positions variance
    for (uint r = 0; r < this->neurons.rows(); r++)
        for (uint c = 0; c < this->neurons.cols(); c++) {
            MatVector<float> dif = average;
            dif.mult(-1);
            dif.add(this->neurons[r][c].weights);
            dif.vabs();
            dif.vqrt();
            variances.add(dif);
        }
}

#endif //DSSOMC_CPP
