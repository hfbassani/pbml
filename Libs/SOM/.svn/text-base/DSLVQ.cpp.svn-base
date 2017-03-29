/* 
 * File:   DSLVQ.cpp
 * Author: flavia
 * 
 * Created on 28 de Fevereiro de 2012, 21:03
 */

#include "DSLVQ.h"
#define maxOf(x,y) (x>y?x:y)

void DSLVQ::initializeClass(vector<int> classes) {

    int c = 0;
    for (int x = 0; x < neurons.rows(); x++)
        for (int y = 0; y < neurons.cols(); y++) {

            neurons[x][y].setClass(classes[c]);
            neurons[x][y].setWon(true);

            c++;
            if (c >= classes.size())
                c = 0;
        }

    /*
    //0: [ 0.77 0.75 ]
    //1: [ 0.26 0.76 ]
    //2: [ 0.50 0.25 ]
    
    neurons[0][0].weights[0][0] = 0.26;
    neurons[0][0].weights[0][1] = 0.76;
    
    neurons[0][1].weights[0][0] = 0.50;
    neurons[0][1].weights[0][1] = 0.25;
    
    neurons[0][2].weights[0][0] = 0.77;
    neurons[0][2].weights[0][1] = 0.75;
    /**/
}

void DSLVQ::updateNeuron(LVQNeuron &neuron, MatVector<float> &data, float alpha, float phi, int dataClass) {

    if (neuron.getClass() == dataClass || !neuron.getWon()) {
        for (uint i = 0; i < data.size(); i++) {
            float distance = fabs(data[i] - neuron.weights[i]);
            neuron.avgDistance[i] = phi * beta_t * distance + (1 - phi * beta_t) * neuron.avgDistance[i];
        }
    }
    for (uint i = 0; i < data.size(); i++) {
        float distance = fabs(data[i] - neuron.weights[i]);
        neuron.avgDistanceN[i] = phi * beta_t * distance + (1 - phi * beta_t) * neuron.avgDistanceN[i];
    }

    float max = neuron.avgDistance.max();
    float maxN = neuron.avgDistanceN.max();

    //update neuron fs weights
    for (uint i = 0; i < data.size(); i++) {

        if (max != 0)
            neuron.dsWeights[i] = 1 - (neuron.avgDistance[i] / max);
        else
            neuron.dsWeights[i] = 1;

        //            if ((neuron.getClass() != dataClass) && maxN != 0)
        //                neuron.dsWeights[wr][wc] = neuron.dsWeights[wr][wc] * (neuron.avgDistanceN[wr][wc] / maxN);

        //neuron.dsWeights[wr][wc] = avgDistGlobal[wr][wc] / (neuron.avgDistance[wr][wc] + 0.000001);

        if (neuron.dsWeights[i] < parameters.minDW)
            neuron.dsWeights[i] = parameters.minDW;
    }

//    max = neuron.dsWeights.max();
//
//    //normalize relevance neurons
//    for (uint i = 0; i < data.size(); i++) {
//        neuron.dsWeights[i] = neuron.dsWeights[i] / max;
//    }

    //update neuron weights
    for (uint i = 0; i < data.size(); i++) {
        if (neuron.getClass() == dataClass || !neuron.getWon())
            neuron.weights[i] = neuron.weights[i] + phi * alpha * (data[i] - neuron.weights[i]);
        else
            neuron.weights[i] = neuron.weights[i] - phi * (alpha*0.75) * (data[i] - neuron.weights[i]);
    }
}

void DSLVQ::updateNeighbor(LVQNeuron &neuron, MatVector<float> &data, float alpha, float phi, int dataClass) {

    /*
    if (neuron.getClass() == dataClass || !neuron.getWon()) {
        for (uint wr = 0; wr < data.rows(); wr++) //for each row weight of data
            for (uint wc = 0; wc < data.cols(); wc++) //for each col weight of data
            {
                float distance = fabs(data[wr][wc] - neuron.weights[wr][wc]);
                neuron.avgDistance[wr][wc] = phi * beta_t * distance + (1 - phi * beta_t) * neuron.avgDistance[wr][wc];
            }
    } else {
        for (uint wr = 0; wr < data.rows(); wr++) //for each row weight of data
            for (uint wc = 0; wc < data.cols(); wc++) //for each col weight of data
            {
                float distance = fabs(data[wr][wc] - neuron.weights[wr][wc]);
                neuron.avgDistanceN[wr][wc] = phi * beta_t * distance + (1 - phi * beta_t) * neuron.avgDistanceN[wr][wc];
            }
    }

    float max = neuron.avgDistance.max();
    float maxN = neuron.avgDistanceN.max();

    //update neuron fs weights
    for (uint wr = 0; wr < data.rows(); wr++) //for each row weight of data
        for (uint wc = 0; wc < data.cols(); wc++) //for each col weight of data
        {
            if (max != 0)
                neuron.dsWeights[wr][wc] = (1 - (neuron.avgDistance[wr][wc] / max));
            else
                neuron.dsWeights[wr][wc] = 1;

            if (maxN != 0)
                neuron.dsWeights[wr][wc] = neuron.dsWeights[wr][wc] * (neuron.avgDistanceN[wr][wc] / maxN);


            if (neuron.dsWeights[wr][wc] < parameters.minDW)
                neuron.dsWeights[wr][wc] = parameters.minDW;
        }
    /**/

    //update neuron weights
    for (uint i = 0; i < data.size(); i++) {
        //if (neuron.getClass() == dataClass || !neuron.getWon())
            neuron.weights[i] = neuron.weights[i] + phi * alpha * (data[i] - neuron.weights[i]);
        //else
        //    neuron.weights[i] = neuron.weights[i] - phi * (alpha/2) * (data[i] - neuron.weights[i]);
    }
}

void DSLVQ::updateNeigbors(LVQNeuron &neuron, MatVector<float> &data, float alpha, int dataClass) {

    uint rmin = max((int) neuron.r - (int) parameters.numNeighbors, 0);
    uint rmax = min((int) neuron.r + (int) parameters.numNeighbors, (int) neurons.rows() - 1);
    uint cmin = max((int) neuron.c - (int) parameters.numNeighbors, 0);
    uint cmax = min((int) neuron.c + (int) parameters.numNeighbors, (int) neurons.cols() - 1);

    for (uint nr = rmin; nr <= rmax; nr++) //for each row neighbor of neuron
        for (uint nc = cmin; nc <= cmax; nc++) //for each col neighbor of neuron
        {
            if (nr == neuron.r && nc == neuron.c)
                updateNeuron(neurons[nr][nc], data, alpha, phi((int) neuron.r - (int) nr, (int) neuron.c - (int) nc), dataClass);
            else
                updateNeighbor(neurons[nr][nc], data, alpha, phi((int) neuron.r - (int) nr, (int) neuron.c - (int) nc), dataClass);
        }
}

void DSLVQ::train(MatMatrix<float> &trainingData, std::vector<int> dataClasses) {
    LVQNeuron bmu;

    for (resetTime(); t < parameters.tmax; incTime()) {
        for (uint i = 0; i < trainingData.rows(); i++) {
            int v = (rand() / (double) RAND_MAX)*(trainingData.rows() - 1);
            MatVector<float> sample;
            trainingData.getRow(v, sample);

            findBMU(sample, bmu);

            if (!bmu.getWon()) {
                neurons[bmu.r][bmu.c].setClass(dataClasses[v]);
                neurons[bmu.r][bmu.c].setWon(true);
            }

            updateNeigbors(bmu, sample, alpha_t, dataClasses[v]);
        }

        if (callback != NULL)
            if (callback(*this)) break;
    }
}

void DSLVQ::trainStep(MatMatrix<float>&trainingData, std::vector<int> dataClasses) {
    LVQNeuron bmu;

    for (uint i = 0; i < trainingData.rows(); i++) {
        int v = (rand() / (double) RAND_MAX)*(trainingData.rows() - 1);
        MatVector<float> sample;
        trainingData.getRow(v, sample);

        findBMU(sample, bmu);

        if (!bmu.getWon()) {
            neurons[bmu.r][bmu.c].setClass(dataClasses[v]);
            neurons[bmu.r][bmu.c].setWon(true);
        }

        updateNeigbors(bmu, sample, alpha_t, dataClasses[v]);
        //        dbgOut(0) << "w" << bmu.c <<":\t" << neurons[0][bmu.c].weights.toString() << ": " << bmu.getClass() << endl;

        if (callback != NULL)
            if (callback(*this)) break;
    }
}

void DSLVQ::getMaxRelevances(MatVector<float> &maxRelevances) {
    maxRelevances.size(inputSize);
    maxRelevances.fill(0);

    for (uint r = 0; r < neurons.rows(); r++)
        for (uint c = 0; c < neurons.cols(); c++) {

            //Compute max relevance for each dimension
            for (uint i = 0; i < maxRelevances.size(); i++) {
                    if (maxRelevances[i] < neurons[r][c].dsWeights[i])
                        maxRelevances[i] = neurons[r][c].dsWeights[i];
                }

        }
}

void DSLVQ::getRelevances(MatVector<float> &relevances) {

    relevances.size(inputSize);
    relevances.fill(0);

    MatVector<float> average(relevances.size());
    average.fill(0);

    for (uint r = 0; r < neurons.rows(); r++)
        for (uint c = 0; c < neurons.cols(); c++) {

            //Compute max relevance for each dimension
            for (uint i = 0; i < relevances.size(); i++) {
                if (relevances[i] < neurons[r][c].dsWeights[i])
                    relevances[i] = neurons[r][c].dsWeights[i];
            }

            //Compute average neurons positions
            average.add(neurons[r][c].weights);
        }

    average.mult(1 / (float) neurons.size());

    MatVector<float> variance(relevances.size());
    variance.fill(0);

    //Compute neuron positions variance
    for (uint r = 0; r < neurons.rows(); r++)
        for (uint c = 0; c < neurons.cols(); c++) {
            MatVector<float> dif = average;
            dif.mult(-1);
            dif.add(neurons[r][c].weights);
            dif.vabs();
            dif.vqrt();
            variance.add(dif);
        }

    //Compute relevances
    for (uint i = 0; i < inputSize; i++) {
        relevances[i] = relevances[i] * variance[i];
    }

    relevances.mult(1 / relevances.max());
}

void DSLVQ::getVariances(MatVector<float> &variances) {

    MatVector<float> average(inputSize);
    average.fill(0);

    for (uint r = 0; r < neurons.rows(); r++)
        for (uint c = 0; c < neurons.cols(); c++) {
            //Compute average neurons positions
            average.add(neurons[r][c].weights);
        }

    average.mult(1 / (float) neurons.size());


    variances.size(inputSize);
    variances.fill(0);

    //Compute neuron positions variance
    for (uint r = 0; r < neurons.rows(); r++)
        for (uint c = 0; c < neurons.cols(); c++) {
            MatVector<float> dif = average;
            dif.mult(-1);
            dif.add(neurons[r][c].weights);
            dif.vabs();
            dif.vqrt();
            variances.add(dif);
        }
}

float DSLVQ::presentPaternToNeuron(const MatVector<float> &data, LVQNeuron &neuron) {
    float distance = 0;

    for (uint i = 0; i < data.size(); i++) {
        distance += neuron.dsWeights[i] * qrt(fabs(data[i] - neuron.weights[i])); //Weighted euclidean
        //distance += neuron.dsWeights[i] * (fabs(data[i] - neuron.weights[i])); //Weighted Mahattan
    }

    //    neuron.dsWeights.mqrt();
    //    float sum = neuron.dsWeights.sum();
    //    distance = distance/(1+sqrt(sum));

    //    distance = distance/(1+neuron.dsWeights.sum());
    //    return 1/(distance+1);

    return (neuron.dsWeights.sum() / (neuron.dsWeights.sum() + distance + 0.0000001));
}
