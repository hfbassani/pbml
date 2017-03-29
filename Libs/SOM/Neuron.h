/* 
 * File:   Neuron.h
 * Author: hans
 *
 * Created on 6 de Outubro de 2010, 17:05
 */

#ifndef _NEURON_H
#define	_NEURON_H

#include "Defines.h"
#include "MatMatrix.h"
#include "Parameters.h"

class Neuron {

  public:
    uint r;
    uint c;
    MatVector<float> weights;
    bool canWin;
    uint wl;
    uint wr;
    float activation;
    float maxActivation;

    Neuron();
    Neuron(uint r, uint c);
    virtual ~Neuron();

    virtual void setupWeights(uint inputSize);
    void set(uint r, uint c);
    void set(uint r, uint c, float activation);
    
    void write(std::ofstream &file) {
        for (int i=0; i<weights.size(); i++) {
            file << weights[i];
            if (i<weights.size()-1) 
                file << "\t";
        }
        file << std::endl;
        
    }
    
    void read(std::istream &file) {
        
        std::string line;
        getline(file, line);
        std::stringstream parserW(line);
        float value;
        int i=0;
        while (!parserW.eof()) {
            parserW >> value;
            if (i<weights.size())
                weights[i] = value;
            else
                weights.append(value);
            i++;
        }
    }
};

std::ostream& operator << (std::ostream& out, Neuron &neuron);
std::istream& operator >> (std::istream& in, Neuron &neuron);

template <class T> bool getValue(const std::string &line, T &value)
{
		std::string valueStr;
		if (getValueFromLine(line, valueStr))
		{
			std::istringstream ist(valueStr);
			ist >> value;
			return true;
		}

		return false;
}

#endif	/* _NEURON_H */

