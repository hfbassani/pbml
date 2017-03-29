/* 
 * File:   DSSOM.h
 * Author: hans
 *
 * Created on 6 de Outubro de 2010, 17:41
 */

#ifndef _DSSOMC_H
#define	_DSSOMC_H

#include "SOM2D.h"
#include "DSNeuron.h"

class DSSOMCParameters: public SOMParameters {

public:

    Parameter<float> beta; //Feature selection learning rate
    Parameter<float> minDW; //Minimum dimension weight
    Parameter<uint> numWinners;  //Number of winners per pattern
    Parameter<float> epsilonRho;  //Number of winners per pattern
    Parameter<float> outliersThreshold; //Outliers threshold

    DSSOMCParameters():SOMParameters() {

            //Comentários do início do arquivo
            comments = "Dimension Selective Self-Organizing Map Parameters";

            //Definição do nome da sessão do arquivo na qual estes parâmetros serão salvos
            section = "DSSOMParameters";

            //Parâmetros persistentes
            addParameterD(beta, "Distance average rate");
            addParameterD(minDW, "Minimum dimension weight");
            addParameterD(epsilonRho, "Minimum relevance required to search for another winner");
            addParameterD(outliersThreshold, "Outliers threshold");

            //Default values
            numNeighbors    = 1;     //Num numNeighbors: City block distance to the farthest neigbors
            alpha           = 0.01; //Alpha0: Learning rate
            beta            = 0.01; //Beta0: Distance average rate
            lambda2         = 0.2;  //Lambda: Learning rate decay with time
            sigma           = 0.3;  //Sigma0: Neighborhood function standard deviation
            lambda1         = 1;    //Lambda: Learning rate decay with time
            tmax            = 1000; //Maximum number of iterations
            minDW           = 0.1;  //Minimum dimension weight
            numWinners      = 5;    //Number of winners per pattern
            epsilonRho      = 0.5;   //Minimum relevance required to search for another winner
            outliersThreshold = 0.90; //Outliers threshold
    }
};

class DSCNeuron: public DSNeuron {
public:
    MatVector<float> avgDistanceN;
    
    DSCNeuron(){};
    DSCNeuron(uint r, uint c):DSNeuron(r,c){};
    void setupWeights(uint inputSize)
    {
        DSNeuron::setupWeights(inputSize);

        avgDistanceN.size(inputSize);
        avgDistanceN.fill(0);
    }
};

template <class TNeuron = DSCNeuron, class TParameters = DSSOMCParameters>
class DSSOMC: public SOM2D<TNeuron, TParameters> {

protected:
    float beta_t;

public:
    MatVector<float> fixed;
    DSSOMC();
    virtual void initialize(uint somRows, uint somCols, uint inputSize);
    void initClasses();
    virtual void reRandomizeWeights();

    void setParameters(uint numNeighbors, float alpha, float beta, float lambda2, float sigma, float lambda1, uint tmax);
    virtual void updateNeuron(TNeuron &neuron, MatVector<float> &data, float alpha, float phi);
    virtual float presentPaternToNeuron(const MatVector<float> &data, TNeuron &neuron);
    virtual bool incTime();
    virtual void resetTime();
    virtual void findBMU_NC(int classW, TNeuron &bmuNC);
    virtual void getVariances(MatVector<float> &variances);
    virtual void getMaxRelevances(MatVector<float> &maxRelevances);
    virtual void getRelevances(MatVector<float> &relevances);

private:

};

#include "DSSOMC.cpp"
#endif	/* _DSSOM_HC */

