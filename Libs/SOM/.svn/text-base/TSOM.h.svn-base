#ifndef TSOM_H
#define TSOM_H

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include "MatMatrix.h"
#include "Parameters.h"
#include "SOM2D.h"


class TNeuron {

  public:
    uint r;
    uint c;
    MatMatrix<float> weights;
    bool canWin;
    uint wl;
    uint wr;
    float maxActivation;

    TNeuron();
    TNeuron(uint r, uint c);
    virtual ~TNeuron();

    void setupWeights(uint wr, uint wc);
    void set(uint r, uint c);
};

typedef MatMatrix<float> FeaturesVector;

class TSOM: public CFGFileObject
{
    uint inputRows;
    uint inputCols;
    MatMatrix<TNeuron> neurons;
    MatMatrix<float> activationMap;
    uint t;

    public:
        SOMParameters parameters;

        TSOM();
        virtual ~TSOM();

        void initialize(uint somRows, uint somCols, uint inputRows, uint inputCols);
        void reRandomizeWeights();
        void setParameters(uint numNeighbors, float alpha, float lambda2, float sigma, float lambda1, uint tmax);
        void train(std::vector<FeaturesVector *> &trainingData);
        void train2(std::vector<FeaturesVector *> &trainingData);
        void trainPhrase(FeaturesVector &featuresVector);
        MatMatrix<float> *updateActivationMap(FeaturesVector &data);
        MatMatrix<float> *getActivationMap();
        std::string toString();
        void fromString(const std::string str);
        MatMatrix<TNeuron> *getNeurons();

        bool incTime();
        void resetTime();

    protected:
    private:
        void presentPaternToSOM(MatMatrix<float> &data);
        float presentPaternToNeuron(MatMatrix<float> &data, TNeuron &neuron);
        float presentPaternToNeuronLRW(MatMatrix<float> &data, TNeuron &neuron, int d, uint &wl, uint &wr);
        TNeuron *findBMU(MatMatrix<float> &data, std::vector<TNeuron *> &winners);
        TNeuron *findBMU3(MatMatrix<float> &data, TNeuron *bmu, int d);
        void updateNeigbors(TNeuron &neuron, MatMatrix<float> &data, float alpha);
        void updateNeuron(TNeuron &neuron, MatMatrix<float> &data, float alpha, float phi);
        float diffToActivation(float diff);
        float phi(float dr, float dc);

        //functions under evaluation
        void presentPaternToSOM2(MatMatrix<float> &data, int d);
        void updateNeigbors2(TNeuron &neuron, MatMatrix<float> &data, float alpha, int d);
        void updateNeuron2(TNeuron &neuron, MatMatrix<float> &data, float alpha, float phi, int d);
        TNeuron *findBMU2(MatMatrix<float> &data, int d);


        std::ostream& toStream(std::ostream& out);
        std::istream& fromStream(std::istream& in);

};

std::ostream& operator << (std::ostream& out, TNeuron &neuron);
std::istream& operator >> (std::istream& in, TNeuron &neuron);


#endif // TSOM_H
