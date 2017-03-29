/* 
 * File:   SOMAD.h
 * Author: hans
 *
 * Created on 27 de Março de 2012, 17:41
 */

#ifndef _SOMAW_H
#define	_SOMAW_H

#include "SOM2D.h"
#include "DSNeuron.h"

class SOMAWParameters: public SOMParameters {

public:

    Parameter<float> k1; 
    Parameter<float> k2; 
    Parameter<float> k3; 

    SOMAWParameters():SOMParameters() {

            //Comentários do início do arquivo
            comments = "Adaptive Weighting SOM (Kangas 1990) Parameters";

            //Definição do nome da sessão do arquivo na qual estes parâmetros serão salvos
            section = "SOMADParameters";

            //Parâmetros persistentes
            addParameterD(k1, "Correction factor 1");
            addParameterD(k2, "Correction factor 2");
            addParameterD(k3, "Correction factor 3");

            //Default values
            numNeighbors    = 1;     //Num numNeighbors: City block distance to the farthest neigbors
            alpha           = 0.01; //Alpha0: Learning rate
            lambda2         = 0.2;  //Lambda: Learning rate decay with time
            sigma           = 0.3;  //Sigma0: Neighborhood function standard deviation
            lambda1         = 1;    //Lambda: Learning rate decay with time
            tmax            = 1000; //Maximum number of iterations
            k1              = 0.0001;
            k2              = 0.99;
            k3              = 1.02;
    }
};

template <class TNeuron = DSNeuron, class TParameters = SOMAWParameters>
class SOMAW: public SOM2D<TNeuron, TParameters> {

protected:
    float beta_t;

public:
    SOMAW();
    virtual void initialize(uint somRows, uint somCols, uint inputSize);
    virtual void reRandomizeWeights();

    void setParameters(uint numNeighbors, float alpha, float lambda2, float sigma, float lambda1, uint tmax, float k1, float k2, float k3);
    virtual void updateNeuron(TNeuron &neuron, MatVector<float> &data, float alpha, float phi);
    virtual float presentPaternToNeuron(const MatVector<float> &data, TNeuron &neuron);

    virtual bool incTime();
    virtual void resetTime();

private:

};

#include "SOMAW.cpp"
#endif	/* _SOMAW_H */

