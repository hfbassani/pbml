/* 
 * File:   TDSSOM.h
 * Author: hans
 *
 * Created on 9 de Novembro de 2010, 14:48
 */


#ifndef TDSSOM_H
#define	TDSSOM_H

#include "DSSOM.h"
#include "Neuron.h"


class MatrixNeuron: public Neuron {
public:
    uint r, c;
    float activation;
    MatrixNeuron(){};
    MatrixNeuron(uint r, uint c) {this->r = r; this->c = c;};
    MatMatrix<float> weights;
    MatMatrix<float> avgDistance;
    MatMatrix<float> dsWeights;
    void setupWeights(uint inputSize, uint inputCols=1);
    MatVector<float> &getDSWeights(); 
};

class TDSSOM: public DSSOM<MatrixNeuron> {
    uint inputCols;
public:
    TDSSOM();
    virtual ~TDSSOM();
    virtual void updateNeigbors(MatrixNeuron &neuron, MatMatrix<float>& data, float alpha);
    virtual void updateNeuron(MatrixNeuron &neuron, MatMatrix<float> &data, float alpha, float phi);
    virtual float presentPaternToNeuron(const MatMatrix<float> &data, MatrixNeuron &neuron);
    virtual void presentPaternToSOM(const MatMatrix<float>& data);
    virtual void train(std::vector<MatMatrix<float> *> &trainingData);
    virtual void trainStep(std::vector<MatMatrix<float> *> &trainingData);
    virtual void updateActivationMap(MatMatrix<float> &data);
    virtual float findBMU(MatMatrix<float>& data, MatrixNeuron &bmu);
private:
    int d;
};

std::ostream& operator << (std::ostream& out, const MatrixNeuron &neuron);
std::istream& operator >> (std::istream& in, MatrixNeuron &neuron);

#endif	/* TDSSOM_H */

