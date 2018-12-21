/* 
 * File:   FNeuron.h
 * Author: hans
 *
 * Created on 6 de Outubro de 2010, 17:34
 */

#ifndef _DNEURON_H
#define	_DNEURON_H

#include "Neuron.h" 

class DSNeuron: public Neuron {
public:
    DSNeuron(){};
    DSNeuron(uint r, uint c):Neuron(r,c){};
    MatVector<float> avgDistance;
    MatVector<float> dsWeights;
    virtual void setupWeights(uint inputSize);
    MatVector<float> &getDSWeights();
    
    void getWeightMatrix(MatMatrix<float> &matrix);
private:

};

std::ostream& operator << (std::ostream& out, const DSNeuron &neuron);
std::istream& operator >> (std::istream& in, DSNeuron &neuron);

#endif	/* _DNEURON_H */

