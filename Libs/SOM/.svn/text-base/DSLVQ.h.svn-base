/* 
 * File:   DSLVQ.h
 * Author: hans
 *
 * Created on 28 de Fevereiro de 2012, 21:08
 */

#ifndef DSLVQ_H
#define	DSLVQ_H

#include "SOM2D.h"
#include "LVQNeuron.h"
#include "DSSOM.h"

class DSLVQ : public DSSOM<LVQNeuron, DSSOMParameters> {
public:

    DSLVQ() {
    };

    MatVector<float> avgDistGlobal;

    void initialize(uint somRows, uint somCols, uint inputSize) {
        DSSOM<LVQNeuron, DSSOMParameters>::initialize(somRows, somCols, inputSize);
        avgDistGlobal.size(inputSize);
        avgDistGlobal.fill(0);
    }

    void initializeClass(vector<int> classes);
    virtual void updateNeuron(LVQNeuron &neuron, MatVector<float> &data, float alpha, float phi, int dataClass);
    virtual void updateNeigbors(LVQNeuron &neuron, MatVector<float> &data, float alpha, int dataClass);
    virtual void updateNeighbor(LVQNeuron &neuron, MatVector<float> &data, float alpha, float phi, int dataClass);
    virtual void train(MatMatrix<float> &trainingData, std::vector<int> dataClasses);
    virtual void trainStep(MatMatrix<float> &trainingData, std::vector<int> dataClasses);
    virtual void getVariances(MatVector<float> &variances);
    virtual void getMaxRelevances(MatVector<float> &maxRelevances);
    virtual void getRelevances(MatVector<float> &relevances);
    virtual float presentPaternToNeuron(const MatVector<float> &data, LVQNeuron &neuron);
};

#endif	/* DSLVQ_H */

