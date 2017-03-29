/* 
 * File:   TDSSOM.cpp
 * Author: hans
 * 
 * Created on 9 de Novembro de 2010, 14:48
 */

#include "TDSSOM.h"
#include "Defines.h"
#include <cfloat>
#include <iomanip>

void MatrixNeuron::setupWeights(uint inputSize, uint inputCols) {
    weights.size(inputSize, inputCols);

    weights.random();
    weights = weights.mult(0.1);
    weights = weights.add(0.50);
}

float TDSSOM::presentPaternToNeuron(const MatMatrix<float> &data, MatrixNeuron &neuron)
{
    float distance=0;
    uint w0 = max(0, (int)inputCols-(d+1));
    uint wmax = min(inputCols, (int)inputCols+(data.cols()-d-1));
    uint d0 = max(0, d+1-(int)inputCols);
    uint cd, cw;

    for (uint r=0; r<inputSize; r++)
    {
        for (cw=0; cw<w0; cw++)
             distance += neuron.dsWeights[r][cw]*2;
        for (cw=wmax; cw<inputCols; cw++)
             distance += neuron.dsWeights[r][cw]*2;
    }

    for (cw=w0, cd = d0; cw<wmax; cw++, cd++)
        for (uint r=0; r<inputSize; r++)
        {
            distance += neuron.dsWeights[r][cw]*qrt(data[r][cd] - neuron.weights[r][cw]);
        }
 
//    return neuron.dsWeights.sum()/(distance+neuron.dsWeights.rows()*neuron.dsWeights.cols());
    distance = distance/(0.0001+neuron.dsWeights.sum());
    return 1/(distance+1);
}

void TDSSOM::updateNeuron(MatrixNeuron &neuron, MatMatrix<float> &data, float alpha, float phi)
{
    float distance=0;
    int w0 = max(0, (int)inputCols-(d+1));
    int wmax = min((int)inputCols, (int)inputCols + (int)(data.cols()-d-1));
    int d0 = max(0, (int)d+1-(int)inputCols);
    uint cd, cw;

    //Update avgDistance out of data scope cw<w0
    for (cw=0; cw<w0; cw++)
        for (uint r=0; r<inputSize; r++)
            neuron.avgDistance[r][cw] = phi*beta_t*2 + (1-phi*beta_t)*neuron.avgDistance[r][cw];

    //Update avgDistance out of data scope cw>=wmax
    for (cw=wmax; cw<inputCols; cw++)
        for (uint r=0; r<inputSize; r++)
            neuron.avgDistance[r][cw] = phi*beta_t*2 + (1-phi*beta_t)*neuron.avgDistance[r][cw];

    //Update avgDistance inside data scope w0<=cw<wmax
    for (cw=w0, cd = d0; cw<wmax; cw++, cd++)
        for (uint r=0; r<inputSize; r++)
        {
            distance = fabs(data[r][cd] - neuron.weights[r][cw]);
            //update neuron weights
            neuron.avgDistance[r][cw] = phi*beta_t*distance + (1-phi*beta_t)*neuron.avgDistance[r][cw];
            //totalDifference += neuron.avgDistance[wr][wc];
        }

    //update neuron dsweights
    float max = neuron.avgDistance.max();
    for (cw=0; cw<inputCols; cw++)
    for (uint r=0; r<inputSize; r++)
    {
        //update neuron weights
        if (max!=0) {
            neuron.dsWeights[r][cw] = 1-(neuron.avgDistance[r][cw]/max);
            if (neuron.dsWeights[r][cw]<0.01)
                neuron.dsWeights[r][cw] = 0.01;
        }
        else
            neuron.dsWeights[r][cw] = 1;
    }

    //update neuron weights
    for (cw=w0, cd = d0; cw<wmax; cw++, cd++)
    for (uint r=0; r<inputSize; r++)
    {
        neuron.weights[r][cw] = neuron.weights[r][cw] + phi*alpha*(data[r][cd] - neuron.weights[r][cw]);
    }
}

void TDSSOM::train(std::vector<MatMatrix<float> *> &trainingData)
{
    MatrixNeuron bmu;

    for (resetTime(); t<parameters.tmax; incTime())
    {
        for (uint v=0; v<trainingData.size(); v++)
        {
            int dtotal = trainingData[v]->cols()+(inputCols-1);
            for (d = 0; d<dtotal; d++)
            {
                findBMU(*(trainingData[v]), bmu);
                updateNeigbors(bmu, *(trainingData[v]), alpha_t);
            }
        }

        if (callback!=NULL)
            if (callback(*this)) break;
    }
}

void TDSSOM::trainStep(std::vector<MatMatrix<float> *> &trainingData)
{
    MatrixNeuron bmu;

    for (uint v=0; v<trainingData.size(); v++)
    {
        int dtotal = trainingData[v]->cols()+(inputCols-1);
        for (d = 0; d<dtotal; d++)
        {
          findBMU(*(trainingData[v]), bmu);
          updateNeigbors(bmu, *(trainingData[v]), alpha_t);
        }

        if (callback!=NULL)
            if (callback(*this)) break;
    }
}

void TDSSOM::updateActivationMap(MatMatrix<float>& data) {
    for (uint r=0; r<neurons.rows(); r++)
    for (uint c=0; c<neurons.cols(); c++)
    {
        activationMap[r][c] = 0;
        int dtotal = data.cols()+(inputCols-1);
        for (d = 0; d<dtotal; d++) {
            neurons[r][c].activation = presentPaternToNeuron(data, neurons[r][c]);
            if (neurons[r][c].activation>activationMap[r][c])
                activationMap[r][c] = neurons[r][c].activation;
        }
        neurons[r][c].activation = activationMap[r][c];
    }
    //activationMap.rescale01();
}

void TDSSOM::presentPaternToSOM(const MatMatrix<float>& data) {
    for (uint r = 0; r < neurons.rows(); r++)
        for (uint c = 0; c < neurons.cols(); c++) {
            activationMap[r][c] = neurons[r][c].activation = presentPaternToNeuron(data, neurons[r][c]);
        }
}

float TDSSOM::findBMU(MatMatrix<float>& data, MatrixNeuron &bmu) {
    uint r, c;

    presentPaternToSOM(data);
    activationMap.max(r, c);
    bmu = neurons[r][c];
    return bmu.activation;
}

void TDSSOM::updateNeigbors(MatrixNeuron &neuron, MatMatrix<float>& data, float alpha) {
    uint rmin = max((int) neuron.r - (int) parameters.numNeighbors, 0);
    uint rmax = min((int) neuron.r + (int) parameters.numNeighbors, (int) neurons.rows() - 1);
    uint cmin = max((int) neuron.c - (int) parameters.numNeighbors, 0);
    uint cmax = min((int) neuron.c + (int) parameters.numNeighbors, (int) neurons.cols() - 1);

    for (uint nr = rmin; nr <= rmax; nr++) //for each row neighbor of neuron
        for (uint nc = cmin; nc <= cmax; nc++) //for each col neighbor of neuron
        {
            updateNeuron(neurons[nr][nc], data, alpha, phi((int) neuron.r - (int) nr, (int) neuron.c - (int) nc));
        }
    //dbgOut() << phi(0,0) << "-" << phi(parameters.numNeighbors,0) << ":" << alpha*phi(0,0) << "-" << alpha*phi(parameters.numNeighbors,0) << endl;
}

std::ostream& operator << (std::ostream& ost, const MatrixNeuron &neuron)
{

   return ost;
}

std::istream& operator >> (std::istream& ist, MatrixNeuron &neuron)
{

   return ist;
}