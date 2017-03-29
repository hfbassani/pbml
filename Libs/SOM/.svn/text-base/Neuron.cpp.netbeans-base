/* 
 * File:   Neuron.cpp
 * Author: hans
 * 
 * Created on 6 de Outubro de 2010, 17:05
 */

#include "Neuron.h"

Neuron::Neuron()
{
    set(0,0);
    wl = wr = 0;
    maxActivation = 0;
}

Neuron::Neuron(uint r, uint c)
{
    set(r,c);
}

void Neuron::setupWeights(uint inputSize)
{
    weights.size(inputSize);

    weights.random();
    weights = weights.mult(0.1);
    weights = weights.add(0.50);
}

void Neuron::set(uint r, uint c)
{
   this->r = r;
   this->c = c;
}

void Neuron::set(uint r, uint c, float activation)
{
   this->r = r;
   this->c = c;
   this->activation = activation;
}


std::ostream& operator << (std::ostream& ost, Neuron &neuron)
{
   ost << "inputSize" << CFGFile::delimiterStr << neuron.weights.size() << CFGFile::endValueStr << endl;

   ost << "weights" << CFGFile::delimiterStr << "\t";
   for (uint i = 0; i < neuron.weights.size(); i++) {
           ost << neuron.weights[i] << "\t";
   }

   return ost;
}

std::istream& operator >> (std::istream& ist, Neuron &neuron)
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

   getline(ist, line); //read '\t';

   return ist;
}


Neuron::~Neuron()
{

}

