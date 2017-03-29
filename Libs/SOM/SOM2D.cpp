
#ifndef SOM_CPP
#define SOM_CPP

#include "SOM2D.h"
#include <math.h>
#include <iostream>
#include "Defines.h"
 
using namespace std;

template <class TNeuron, class TSOMParameters>
SOM2D<TNeuron, TSOMParameters>::SOM2D() {
    section = "SOM";
    comments = "Self Organizing Map";
    callback = NULL;
}

template <class TNeuron, class TSOMParameters>
SOM2D<TNeuron, TSOMParameters>::~SOM2D() {
    //dtor
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::setParameters(uint numNeighbors, float alpha, float lambda, float sigma, uint tmax) {
    parameters.numNeighbors = numNeighbors;
    parameters.alpha = alpha;
    parameters.lambda1 = lambda;
    parameters.sigma = sigma;
    parameters.tmax = tmax;
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::initialize(uint somRows, uint somCols, uint inputSize) {
    this->somCols = somCols;
    this->somRows = somRows;
    this->inputSize = inputSize;
    resetTime();

    neurons.size(somRows, somCols);
    activationMap.size(somRows, somCols);

    reRandomizeWeights();
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::initializeP(const TSOMParameters &parameters) {
    this->parameters = parameters;
    initialize(this->parameters.NX, this->parameters.NY, this->parameters.NFeatures);
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::reRandomizeWeights() {

    for (uint r = 0; r < neurons.rows(); r++)
        for (uint c = 0; c < neurons.cols(); c++) {
            neurons[r][c].set(r, c);
            neurons[r][c].setupWeights(inputSize);
        }

    activationMap.fill(0);
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::setupGridWeights(MatVector<float> &center, MatVector<float> &delta) {
    MatVector<float> minusDelta2(delta);
    minusDelta2.mult(-0.5);
    MatVector<float> corner(center);
    corner.add(minusDelta2);

    dbgOut() << "Delta: " << delta.toString() << endl;
    dbgOut() << "-Delta/2: " << minusDelta2.toString() << endl;
    dbgOut() << "Corner: " << corner.toString() << endl;

    for (uint r = 0; r < neurons.rows(); r++)
        for (uint c = 0; c < neurons.cols(); c++) {
            neurons[r][c].set(r, c);


            MatVector<float> shift(delta);
            shift[0] = shift[0] * r / (float) neurons.rows();
            shift[1] = shift[1] * c / (float) neurons.cols();
            dbgOut() << "shift: " << shift.toString() << endl;

            MatVector<float> deltaPos(corner);
            deltaPos.add(shift);

            neurons[r][c].weights = deltaPos;
        }

    activationMap.fill(0);
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::resetTime() {
    t = 0;
    //alpha_t = ((parameters.tmax-t)*parameters.alpha)/parameters.tmax;  //linear decay
    alpha_t = parameters.alpha * exp(-(float) t / (parameters.tmax * parameters.lambda2)); //exponencial decay
    sigma_t = parameters.sigma * exp(-((int) t + 1) / (parameters.tmax * parameters.lambda1));
}

template <class TNeuron, class TSOMParameters>
bool SOM2D<TNeuron, TSOMParameters>::incTime() {
    t++;
    if (t < parameters.tmax) {
        //alpha_t = ((parameters.tmax-t)*parameters.alpha)/parameters.tmax;  //linear decay
        alpha_t = parameters.alpha * exp(-(float) t / (parameters.tmax * parameters.lambda2)); //exponencial decay
        if (alpha_t < 0.00001) alpha_t = 0.00001;
        sigma_t = parameters.sigma * exp(-((int) t + 1) / (parameters.tmax * parameters.lambda1));
        if (sigma_t < 0.000001) sigma_t = 0.000001;
    }

    return t < parameters.tmax;
}

template <class TNeuron, class TSOMParameters>
float SOM2D<TNeuron, TSOMParameters>::getAlpha_t() const {
    return alpha_t;
}

template <class TNeuron, class TSOMParameters>
float SOM2D<TNeuron, TSOMParameters>::getSigma_t() const {
    return sigma_t;
}

template <class TNeuron, class TSOMParameters>
uint SOM2D<TNeuron, TSOMParameters>::getT() const {
    return t;
}

template <class TNeuron, class TSOMParameters>
TSOMParameters SOM2D<TNeuron, TSOMParameters>::getParameters() const {
    return parameters;
}

template <class TNeuron, class TSOMParameters>
uint SOM2D<TNeuron, TSOMParameters>::getInputSize() const {
    return inputSize;
}

template <class TNeuron, class TSOMParameters>
uint SOM2D<TNeuron, TSOMParameters>::getSomCols() const {
    return somCols;
}

template <class TNeuron, class TSOMParameters>
uint SOM2D<TNeuron, TSOMParameters>::getSomRows() const {
    return somRows;
}

template <class TNeuron, class TSOMParameters>
float SOM2D<TNeuron, TSOMParameters>::presentPaternToNeuron(const MatVector<float> &data, TNeuron &neuron) {
    float distance = 0;

    for (uint i = 0; i < data.size(); i++) {
        distance += qrt(fabs(data[i] - neuron.weights[i]));
    }

    return 1 / (distance + 1);
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::presentPaternToSOM(const MatVector<float> &data) {
    for (uint r = 0; r < neurons.rows(); r++)
        for (uint c = 0; c < neurons.cols(); c++) {
            activationMap[r][c] = neurons[r][c].activation = presentPaternToNeuron(data, neurons[r][c]);
        }
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::getClusterPosition(const MatVector<float> &data, uint &r, uint &c) {
    TNeuron bmu;
    findBMU(data, bmu);
    r = bmu.r;
    c = bmu.c;
}
//Find the best matching neuron

template <class TNeuron, class TSOMParameters>
float SOM2D<TNeuron, TSOMParameters>::findBMU(const MatVector<float> &data, TNeuron &bmu) {
    uint r, c;

    presentPaternToSOM(data);
    activationMap.max(r, c);
    bmu = neurons[r][c];
    return bmu.activation;
}

template <class TNeuron, class TSOMParameters>
float SOM2D<TNeuron, TSOMParameters>::phi(float dr, float dc) {
    float d2 = qrt(dr) + qrt(dc);
    //    sigma_t = parameters.sigma*exp(-((int)t+1)/(parameters.tmax*parameters.lambda1));
    float phi = exp(-d2 / (2 * qrt(sigma_t)));

    return phi;
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::updateNeuron(TNeuron &neuron, MatVector<float> &data, float alpha, float phi) {
    for (uint i = 0; i < data.size(); i++)
        neuron.weights[i] = neuron.weights[i] + phi * alpha * (data[i] - neuron.weights[i]);
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::updateNeigbors(TNeuron &neuron, MatVector<float> &data, float alpha) {
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

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::train(MatMatrix<float> &trainingData) {
    TNeuron bmu;

    for (resetTime(); t < parameters.tmax; incTime()) {
        trainStep(trainingData);
        if (callback != NULL)
            if (callback(*this)) break;
    }
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::trainStep(MatMatrix<float> &trainingData) {
    TNeuron bmu;

    int v = (rand() / (double) RAND_MAX)*(trainingData.rows() - 1);
    MatVector<float> sample;
    trainingData.getRow(v, sample);

    findBMU(sample, bmu);
    updateNeigbors(bmu, sample, alpha_t);
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::getNeuronWeights(TNeuron &neuron, MatVector<float> & neuronWeights) {

    neuronWeights = neurons[neuron.r][neuron.c].weights;
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::getNeuron(TNeuron &neuron) {
    neuron = neurons[neuron.r][neuron.c];
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::getNeurons(MatMatrix<TNeuron> &neurons) const {
    neurons = this->neurons;
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::getActivationMap(MatMatrix<float> &activationMap) {
    activationMap = this->activationMap;
}

template <class TNeuron, class TSOMParameters>
void SOM2D<TNeuron, TSOMParameters>::updateActivationMap(MatVector<float> &data) {
    presentPaternToSOM(data);
}

///Persistence functions

template <class TNeuron, class TSOMParameters>
std::ostream& SOM2D<TNeuron, TSOMParameters>::toStream(std::ostream& out) {
    out << "somRows" << CFGFile::delimiterStr << neurons.rows() << CFGFile::endValueStr << endl;
    out << "somCols" << CFGFile::delimiterStr << neurons.cols() << CFGFile::endValueStr << endl;
    out << "inputSize" << CFGFile::delimiterStr << inputSize << CFGFile::endValueStr << endl;

    for (uint r = 0; r < neurons.rows(); r++) {
        for (uint c = 0; c < neurons.cols(); c++)
            out << neurons[r][c] << endl;
    }

    return out;
}

template <class TNeuron, class TSOMParameters>
std::istream& SOM2D<TNeuron, TSOMParameters>::fromStream(std::istream& in) {
    string line;

    while (getline(in, line)) {
        if (checkEndSection(line))
            break;

        if (getValue(line, somRows)) {
            getline(in, line);
            getValue(line, somCols);

            getline(in, line);
            getValue(line, inputSize);

            initialize(somRows, somCols, inputSize);
            for (uint r = 0; r < neurons.rows(); r++) {
                for (uint c = 0; c < neurons.cols(); c++)
                    in >> neurons[r][c];
            }
        }
    }

    return in;
}

#endif //SOM_CPP
