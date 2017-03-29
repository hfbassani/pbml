/* 
 * File:   FNeuron.cpp
 * Author: hans
 * 
 * Created on 6 de Outubro de 2010, 17:34
 */

#include "DSNeuron.h" 

void DSNeuron::setupWeights(uint inputSize)
{
    Neuron::setupWeights(inputSize);

    avgDistance.size(inputSize);
    avgDistance.fill(0);
    dsWeights.size(inputSize);
    dsWeights.fill(1);
}

MatVector<float> &DSNeuron::getDSWeights() {
    return dsWeights;
}

void DSNeuron::getWeightMatrix(MatMatrix<float> &matrix) {
    int rows = dsWeights.size()/matrix.cols();
    matrix.size(rows, matrix.cols());
            
    uint r=0;
    uint c=0;
    for (uint v = 0; v<dsWeights.size(); v++) {
        matrix[r][c] = dsWeights[v];
        if (c==matrix.cols()) {
            c = 0;
            r++;
        }
    }
}

std::ostream& operator << (std::ostream& ost, const DSNeuron &neuron)
{
   ost << "size" << CFGFile::delimiterStr << neuron.weights.size() << CFGFile::endValueStr << endl;

   ost << "weights" << CFGFile::delimiterStr << "\t";
   for (uint i = 0; i < neuron.weights.size(); i++) {
           ost << neuron.weights[i] << "\t";
   }

   for (uint i = 0; i < neuron.weights.size(); i++) {
           ost << neuron.dsWeights[i] << "\t";
   }

   return ost;
}

std::istream& operator >> (std::istream& ist, DSNeuron &neuron)
{
   uint size;
   std::string line;

   getline(ist, line);
   getValue(line, size);

   neuron.weights.size(size);
   ist >> line; //read "weights:";

   for (uint i = 0; i < neuron.weights.size(); i++) {
           ist >> neuron.weights[i];
   }

   for (uint i = 0; i < neuron.weights.size(); i++) {
           ist >> neuron.dsWeights[i];
   }

   getline(ist, line); //read '\t';

   return ist;
}