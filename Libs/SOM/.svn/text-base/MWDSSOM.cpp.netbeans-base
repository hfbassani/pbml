
/* 
 * File:   FSOM.cpp
 * Author: hans
 * 
 * Created on 6 de Outubro de 2010, 17:41
 */

#include "MWDSSOM.h"
#include "DebugOut.h"
//#include "Perceptions.h"
#include <math.h>

MWDSSOM::MWDSSOM() {
    section = "MWDSSOM";
    comments = "Multiple Winners Dimension Selective Self-Organizing Map";
    callback = NULL;
}

void MWDSSOM::initialize(uint somRows, uint somCols, uint inputSize)
{
    this->somRows = somRows;
    this->somCols = somCols;
    this->inputSize = inputSize;
    resetTime();

    neurons.size(somRows, somCols);
    activationMap.size(somRows, somCols);
    relevance.size(inputSize);
    relevance.fill(1);

    reRandomizeWeights();
}

void MWDSSOM::resetRelevances() {
    relevance.fill(1); //Set all features as full relevant
}

void MWDSSOM::updateRelevances(DSNeuron &bmu) {
   //Reduce the relevance of features already considered by current bmu
   for (uint i = 0; i < relevance.size(); i++)
      relevance[i] = relevance[i]*(1-bmu.dsWeights[i]);
      //relevance[r][c] = relevance[r][c]*exp(-bmu.dsWeights[r][c]*7);
}

const MatVector<float> MWDSSOM::getRelevance() {
    return relevance;
}

float MWDSSOM::findFirstBMU(const MatVector<float> &data, DSNeuron &bmu) {
       
    uint r,c;
    for (r=0; r<neurons.rows(); r++)
    for (c=0; c<neurons.cols(); c++)
        neurons[r][c].canWin = true;

    presentPaternToSOM(data);

    activationMap.max(r,c);
    bmu = neurons[r][c];

    return bmu.activation;
}

float MWDSSOM::findNextBMU(const MatVector<float> &data, DSNeuron &bmu) {

    neurons[bmu.r][bmu.c].canWin = false; //A winner neuron can not win again for this input

    uint r,c;
    /*     
    for (r=0; r<neurons.rows(); r++)
    for (c=0; c<neurons.cols(); c++)
    {
        //activationMap[r][c] = neurons[r][c].activation = presentPaternToNeuronAgain(data, neurons[r][c]);
        activationMap[r][c] = neurons[r][c].activation = presentPaternToNeuron(data, neurons[r][c]);
    }/**/

    while (true) {
     activationMap.max(r,c);
     bmu = neurons[r][c];

     if (activationMap[r][c] == 0)
         break;

     if (bmu.canWin)
         break;
     else
         activationMap[r][c] = 0;
    };

    return bmu.activation;
}

float MWDSSOM::findNextBMU(DSNeuron &bmu)
{
    uint r,c;
    float difference;
    DSNeuron prevBmu = bmu;
    MatMatrix<float> dsWeights;

//    dbgOut(1) << "Winner0 [" << bmu.r << "," << bmu.c << "]: " << featuresToString(bmu.dsWeights) << endl;
    do {
        //desativa vencedor atual
        activationMap[bmu.r][bmu.c] = neurons[bmu.r][bmu.c].activation = 0;
        //pega o próximo vencedor
        activationMap.max(r,c);
        bmu = neurons[r][c];

        //Computa a distancia do pesso do vencedor atual para o vencedor original
        dsWeights = bmu.dsWeights;
        dsWeights.mult(-1);
        dsWeights.add(prevBmu.dsWeights);
        dsWeights.mqrt();
        difference = dsWeights.sum()/(dsWeights.rows()*dsWeights.cols());

        //Se a distancia > 0.2 encontrou vencedor de outro grupo
        if (difference>0.2) break;

        //Finaliza a busca se encontrar algum vencedor com ativação igual a zero
      } while (bmu.activation>0);
    //} while (bmu.activation>0.5*prevBmu.activation);

 //   if (bmu.activation > 0.5*prevBmu.activation)
       // dbgOut(1) << "WinnerX [" << bmu.r << "," << bmu.c << "]: "  << featuresToString(bmu.dsWeights) << endl;

      return bmu.activation;
}

float MWDSSOM::presentPaternToNeuronAgain(const MatVector<float> &data, DSNeuron &neuron)
{
    float distance = 0;

    for (uint i = 0; i < data.size(); i++) {
            distance += (neuron.dsWeights[i]*relevance[i])*qrt(fabs(data[i] - neuron.weights[i]));
        }
    
    /**/
    float sumq=0;
    for (uint i = 0; i < data.size(); i++) {

            sumq += qrt(neuron.dsWeights[i]-relevance[i]);
        }
    sumq = qrt(sumq);
    return (sumq/(sumq+distance+0.0000001));
    /**/
    
    //return (neuron.dsWeights.sum()/(neuron.dsWeights.sum()+distance*distance+0.0000001));
}

float MWDSSOM::presentPaternToNeuron(const MatVector<float> &data, DSNeuron &neuron)
{
    float distance = 0;

    for (uint i = 0; i < data.size(); i++) {
            distance += (neuron.dsWeights[i])*qrt(fabs(data[i] - neuron.weights[i]));
        }
    
    /**/
    float sumq=0;
    for (uint i = 0; i < data.size(); i++) {
            sumq  += fabs(neuron.dsWeights[i]);
        }
    sumq = sqrt(sumq);
    //return (sumq/(sumq+distance+0.0000001));
    return (exp(-(distance)));
    /**/
    
    //return (neuron.dsWeights.sum()/(neuron.dsWeights.sum()+distance*distance+0.0000001));
}

void MWDSSOM::deactivateNeighbors(DSNeuron &neuron)
{
    uint rmin = max((int)neuron.r-(int)parameters.numNeighbors, 0);
    uint rmax = min((int)neuron.r+(int)parameters.numNeighbors, (int)neurons.rows()-1);
    uint cmin = max((int)neuron.c-(int)parameters.numNeighbors, 0);
    uint cmax = min((int)neuron.c+(int)parameters.numNeighbors, (int)neurons.cols()-1);

    for (uint nr = rmin; nr<=rmax; nr++) //for each row neighbor of neuron
    for (uint nc = cmin; nc<=cmax; nc++) //for each col neighbor of neuron
    {
        activationMap[nr][nc] = neurons[nr][nc].activation = 0;
    }
}

void MWDSSOM::train(MatMatrix<float> &trainingData)
{
    DSNeuron bmu;

    for (resetTime(); t<parameters.tmax; incTime())
    {
        trainStep(trainingData);
        if (callback!=NULL)
            if (callback(*this)) break;
    }
    relevance.fill(1);
}

void MWDSSOM::trainStep(MatMatrix<float> &trainingData)
{
    DSNeuron bmu;

    for (uint i=0; i<trainingData.rows(); i++)
    {
        resetRelevances();
        
        int v = (rand() / (double) RAND_MAX)*(trainingData.rows() - 1);
        MatVector<float> sample;
        trainingData.getRow(v, sample);
        findFirstBMU(sample, bmu);

        updateNeigbors(bmu, sample, alpha_t);

        float alpha_t2 = alpha_t;
        for (uint w=1; w<parameters.numWinners; w++)
        {
           updateRelevances(bmu);
           if (relevance.max()<=parameters.epsilonRho) break;
           alpha_t2 = alpha_t2*sigma_t/parameters.sigma;

           findNextBMU(sample, bmu);

           /*float max=0;
           for (int r=0; r< relevance.cols(); r++) {
               float rds = relevance[0][r]*bmu.dsWeights[0][r];
               if (rds>max)
                   max = rds;
           }
           if (max>parameters.epsilonRho)*/
           float beta_t2 = beta_t;
           beta_t = 0;
           updateNeigbors(bmu, sample, alpha_t2);
           beta_t = beta_t2;
        }

        //dbgOut(1) << i << ": " << relevance.max() << endl;
        
        if (callback!=NULL)
            if (callback(*this)) break;

        //dbgOut(0) << (*trainingData[v]).toString() << endl;
    }
    resetRelevances();
}
