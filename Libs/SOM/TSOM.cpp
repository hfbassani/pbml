#include "TSOM.h"
#include <math.h>
#include <cfloat>
#include <iostream>

using namespace std;

TSOM::TSOM()
{
    section = "TSOM";
    comments = "Time Self Organizing Map";
}

TSOM::~TSOM()
{
    //dtor
}

void TSOM::setParameters(uint numNeighbors, float alpha, float lambda2, float sigma, float lambda1, uint tmax)
{
    parameters.numNeighbors = numNeighbors;
    parameters.alpha = alpha;
    parameters.lambda2 = lambda2;
    parameters.sigma = sigma;
    parameters.lambda1 = lambda1;
    parameters.tmax = tmax;
}

void TSOM::initialize(uint somRows, uint somCols, uint inputRows, uint inputCols)
{
    this->inputRows = inputRows;
    this->inputCols = inputCols;
    resetTime();

    neurons.size(somRows, somCols);
    activationMap.size(somRows, somCols);

    reRandomizeWeights();
}

void TSOM::reRandomizeWeights()
{
    //srand(time(NULL));

    for (uint r=0; r<neurons.rows(); r++)
    for (uint c=0; c<neurons.cols(); c++)
    {
      neurons[r][c].set(r,c);
      neurons[r][c].setupWeights(inputRows, inputCols);
    }

    activationMap.fill(0);
}

float TSOM::presentPaternToNeuron(MatMatrix<float> &data, TNeuron &neuron)
{
    float diff, mindiff = FLT_MAX;
    int cd = data.cols();
    uint c, w, w2;

    dbgOut(2) << "neuron:" << neuron.r << endl;
    for (int d = 0; d<cd; d++)
    {
        w2 = min((int)neuron.wr+1, cd-d); //último peso ou tamanho que resta da palavra, o que for menor

        diff = 0;
        for (w=0, c = d+w; w<w2; w++, c++)
        {
            if (w>=neuron.wl && w<=neuron.wr)
                for (uint r=0; r<inputRows; r++)
                {
                    diff += qrt(data[r][c] - neuron.weights[r][w]);
                }
            else {
                diff += 2; // Add maximum distance to the components not present at the input pattern
            }
        }

        if (diff<mindiff)
        {
           mindiff = diff;
        }
    }

    return diffToActivation(mindiff/qrt(1+neuron.wr-neuron.wl));
}

float TSOM::presentPaternToNeuronLRW(MatMatrix<float> &data, TNeuron &neuron, int d, uint &wl, uint &wr)
{
    float rdiff, diff, diff_d, mindiff=FLT_MAX, mindiff_d=FLT_MAX;
    int cd = data.cols();
    int cw = inputCols;
    int w, c, midw, maxw;

    maxw = min(cw, cd-d); //último peso ou tamanho que resta da palavra, o que for menor
    midw = wr = wl = cw/2;

    diff = rdiff = 0;
    //compair to right
    for (w = midw, c = d+w; w<maxw; w++, c++)
    {
        for (uint r=0; r<inputRows; r++)
        {
            rdiff += qrt(data[r][c] - neuron.weights[r][w]);
        }

        diff_d = (1+diff+rdiff)/qrt(1+wr-wl);

        if (diff_d<=mindiff_d)
        {
            mindiff_d = diff_d;
            diff+=rdiff;
            mindiff = diff;
            rdiff = 0;
            wr = w;
        }
//        MatVector<float> features;
//        std::string phoneme;
//        data.getCol(c, features);
//        control.translateFeaturesPhoneme(features, phoneme);
//        dbgOut(2) << "d:" << d << " -> " << phoneme << " <=> ";
//        neuron.weights.getCol(w, features);
//        control.translateFeaturesPhoneme(features, phoneme);
//        dbgOut(2) << phoneme << endl;
    }

    //compair to left
    if (midw-1<cd-d)
    for (w = midw-1, c = d+w; w>=0; w--, c--)
    {
        for (uint r=0; r<inputRows; r++)
        {
            rdiff += qrt(data[r][c] - neuron.weights[r][w]);
        }

        diff_d = (1+diff+rdiff)/qrt(1+wr-wl);

        if (diff_d<=mindiff_d)
        {
            mindiff_d = diff_d;
            diff+=rdiff;
            mindiff = diff;
            rdiff = 0;
            wl = w;
        }
//        MatVector<float> features;
//        std::string phoneme;
//        data.getCol(c, features);
//        control.translateFeaturesPhoneme(features, phoneme);
//        dbgOut(2) << "d:" << d << " -> " << phoneme << " <=> ";
//        neuron.weights.getCol(w, features);
//        control.translateFeaturesPhoneme(features, phoneme);
//        dbgOut(2) << phoneme << endl;
    }

    return diffToActivation(mindiff_d);
}

float TSOM::diffToActivation(float diff)
{
    return 1/exp(diff);
}

void TSOM::presentPaternToSOM(MatMatrix<float> &data)
{
    for (uint r=0; r<neurons.rows(); r++)
        for (uint c=0; c<neurons.cols(); c++)
        {
            activationMap[r][c] = presentPaternToNeuron(data, neurons[r][c]);
        }
}

TNeuron *TSOM::findBMU(MatMatrix<float> &data, vector<TNeuron *> &winners)
{
    uint r,c;
    TNeuron *winner;

    presentPaternToSOM(data);

    //Inihbit winners
    for (uint i=0; i< winners.size(); i++)
    {
        winner = winners[i];
        activationMap[winner->r][winner->c] = -1;
    }

    activationMap.max(r,c);

    return &neurons[r][c];
}

void TSOM::updateNeuron(TNeuron &neuron, MatMatrix<float> &data, float alpha, float phi)
{
    int cd = data.cols();
    int cw = inputCols;
    uint c1, c2;
    uint w;
    int d;
    uint wl,wr;

    presentPaternToNeuronLRW(data, neuron, d, wl,wr);

    c1 = max(d-cw+1,0);
    c2 = min(d,cd-1);

    for (uint r=0; r<inputRows; r++)
    {
      for (uint c=c1; c<=c2; c++)
      {
          w = max(cw-d-1+(int)c,0);
          neuron.weights[r][w] = neuron.weights[r][w] + phi*alpha*(data[r][c] - neuron.weights[r][w]);
      }
    }
}

void TSOM::updateNeigbors(TNeuron &neuron, MatMatrix<float> &data, float alpha)
{
    uint rmin = max((int)neuron.r-(int)parameters.numNeighbors, 0);
    uint rmax = min((int)neuron.r+(int)parameters.numNeighbors, (int)neurons.rows()-1);
    uint cmin = max((int)neuron.c-(int)parameters.numNeighbors, 0);
    uint cmax = min((int)neuron.c+(int)parameters.numNeighbors, (int)neurons.cols()-1);

    for (uint nr = rmin; nr<=rmax; nr++) //for each row neighbor of neuron
    for (uint nc = cmin; nc<=cmax; nc++) //for each col neighbor of neuron
    {
        updateNeuron(neurons[nr][nc], data, alpha, phi((int)neuron.r-(int)nr,(int)neuron.c-(int)nc));
    }
}

float TSOM::phi(float dr, float dc)
{
    float d2 = qrt(dr) + qrt(dc);
//  float phi = exp(-d2/(2*qrt(parameters.sigma)*(t+1)));
    float sigmat = parameters.sigma*exp(-((int)t+1)/parameters.tmax*parameters.lambda1);
    float phi = (1-d2/1.5)*exp(-d2/sigmat);

    return phi;
}

void TSOM::train(std::vector<FeaturesVector *> &trainingData)
{
    TNeuron *bmu; //best matching unit
    float alpha_t;
    vector<TNeuron *> winners, empty;
    uint numNeurons = neurons.cols()*neurons.rows();

    for (t=0; t<parameters.tmax; t++)
    {
      alpha_t = parameters.alpha*exp(-(float)t/(parameters.tmax*parameters.lambda2));  //linear decay

      for (uint v=0; v<trainingData.size(); v++)
      {
        if (winners.size()<numNeurons)
        {
          bmu = findBMU(*(trainingData[v]),winners);
          winners.push_back(bmu);
        }
        else
          bmu = findBMU(*(trainingData[v]),empty);

        updateNeigbors(*bmu, *(trainingData[v]), alpha_t);
      }
    }
}

void TSOM::resetTime()
{
   t = 0;
}

bool TSOM::incTime()
{
   if (t<=parameters.tmax)
   {
     t++;
     return true;
   }
   else{
     return false;
   }

}

void TSOM::trainPhrase(FeaturesVector &featuresVector)
{
    TNeuron *bmu = NULL; //best matching unit
    int cd = featuresVector.cols();
    float alpha_t = parameters.alpha*exp(-(float)t/(parameters.tmax*parameters.lambda2));

    activationMap.fill(0);
    for (int d = 0; d<cd; d++)
    {
      //bmu = findBMU3(featuresVector, bmu, d);
      bmu = findBMU2(featuresVector, d);

      if (bmu==NULL)
        dbgOut(2) << "d: "<< d << "> Winner: bmu not found"<< endl;
      else
      {
        dbgOut(2) << "d: "<< d << "> Winner: [" << bmu->r << "," << bmu->c << "]: " << activationMap[bmu->r][bmu->c] << ", [" << bmu-> wl <<  " " << bmu-> wr << "]" << endl;
        updateNeigbors2(*bmu, featuresVector, alpha_t, d);
      }
    }
    dbgOut(2) << endl;

    //incTime();
}

void TSOM::train2(std::vector<FeaturesVector *> &trainingData)
{
    int d, cd;
    TNeuron *bmu = NULL; //best matching unit
    float alpha_t;

    for (t=0; t<parameters.tmax; t++)
    {
      alpha_t = parameters.alpha*exp(-(float)t/(parameters.tmax*parameters.lambda2));  //linear decay

      for (uint v=0; v<trainingData.size(); v++)
      {
          cd = trainingData[v]->cols();

          activationMap.fill(0);
          for (d = 0; d<cd; d++)
          {
              bmu = findBMU2(*(trainingData[v]),d);
              if (bmu!=NULL)
                updateNeigbors2(*bmu, *(trainingData[v]), alpha_t, d);
          }
      }
    }
}

void TSOM::presentPaternToSOM2(MatMatrix<float> &data, int d)
{
    float activation;
    uint wl, wr;

    for (uint r=0; r<neurons.rows(); r++)
        for (uint c=0; c<neurons.cols(); c++)
        {
            activation = presentPaternToNeuronLRW(data, neurons[r][c], d, wl, wr);
            if (activation >= activationMap[r][c])
            {
                activationMap[r][c] = activation;
                neurons[r][c].canWin = true;

                if (activation>neurons[r][c].maxActivation)
                {
                    neurons[r][c].wl = wl;
                    neurons[r][c].wr = wr;
                    neurons[r][c].maxActivation = activation;
                }
                else if (activation==neurons[r][c].maxActivation && (wr-wl)>(neurons[r][c].wr-neurons[r][c].wl))
                {
                    neurons[r][c].wl = wl;
                    neurons[r][c].wr = wr;
                }
            }
            else
                neurons[r][c].canWin = false;
        }
}

TNeuron *TSOM::findBMU2(MatMatrix<float> &data, int d)
{
    uint r,c;

    presentPaternToSOM2(data, d);
    MatMatrix<float> activationTemp(activationMap);

    activationTemp.max(r,c);
    while (!neurons[r][c].canWin)
    {
        if (activationTemp[r][c] == 0)
          return NULL;

        activationTemp[r][c] = -0; // inhibith this neuron
        activationTemp.max(r,c);  // find another bmu
    }

    return &neurons[r][c];
}

TNeuron *TSOM::findBMU3(MatMatrix<float> &data, TNeuron *bmu, int d)
{
    if (bmu==NULL)
    {
        uint r,c;
        presentPaternToSOM2(data, d);

        activationMap.max(r,c);

        return &neurons[r][c];
    }
    else //search only among current bmu neigbors
    {
        TNeuron *newBMU;
        float maxActivation=0;
        uint wl, wr;

        uint rmin = max((int)bmu->r-(int)parameters.numNeighbors, 0);
        uint rmax = min((int)bmu->r+(int)parameters.numNeighbors, (int)neurons.rows()-1);
        uint cmin = max((int)bmu->c-(int)parameters.numNeighbors, 0);
        uint cmax = min((int)bmu->c+(int)parameters.numNeighbors, (int)neurons.cols()-1);

        for (uint nr = rmin; nr<=rmax; nr++) //for each row neighbor of neuron
        for (uint nc = cmin; nc<=cmax; nc++) //for each col neighbor of neuron
        if (nr!=bmu->r && nc!=bmu->c)
        {
            activationMap[nr][nc] = presentPaternToNeuronLRW(data, neurons[nr][nc], d, wl, wr);
            if (activationMap[nr][nc]>maxActivation)
            {
                newBMU = &neurons[nr][nc];
                maxActivation = activationMap[nr][nc];
                neurons[nr][nc].wl = wl;
                neurons[nr][nc].wr = wr;
            }
        }

        return newBMU;
    }
}

void TSOM::updateNeigbors2(TNeuron &neuron, MatMatrix<float> &data, float alpha, int d)
{
    uint rmin = max((int)neuron.r-(int)parameters.numNeighbors, 0);
    uint rmax = min((int)neuron.r+(int)parameters.numNeighbors, (int)neurons.rows()-1);
    uint cmin = max((int)neuron.c-(int)parameters.numNeighbors, 0);
    uint cmax = min((int)neuron.c+(int)parameters.numNeighbors, (int)neurons.cols()-1);

    for (uint nr = rmin; nr<=rmax; nr++) //for each row neighbor of neuron
    for (uint nc = cmin; nc<=cmax; nc++) //for each col neighbor of neuron
    {
        updateNeuron2(neurons[nr][nc], data, alpha, phi((int)neuron.r-(int)nr,(int)neuron.c-(int)nc), d);
        neurons[nr][nc].wl = neuron.wl;
        neurons[nr][nc].wr = neuron.wr;
    }
}

void TSOM::updateNeuron2(TNeuron &neuron, MatMatrix<float> &data, float alpha, float phi, int d)
{
    int cd = data.cols();
    int cw = inputCols;
    uint c, w, w2;

    w2 = min(cw, cd-d); //último peso ou tamanho que resta da palavra, o que for menor
    for (c = d, w=0; w<w2; w++, c++)
    {
//        MatVector<float> features;
//        std::string phoneme;
//        neuron.weights.getCol(w, features);
//        control.translateFeaturesPhoneme(features, phoneme);
//        dbgOut(2) << "d:" << d << " -> " << phoneme << " <= ";
//        data.getCol(c, features);
//        control.translateFeaturesPhoneme(features, phoneme);
//        dbgOut(2) << phoneme << endl;

        for (uint r=0; r<inputRows; r++)
        {
          neuron.weights[r][w] = neuron.weights[r][w] + phi*alpha*(data[r][c] - neuron.weights[r][w]);
        }
    }
}

MatMatrix<float> *TSOM::getActivationMap()
{
    return &activationMap;
}

MatMatrix<float> *TSOM::updateActivationMap(FeaturesVector &data)
{
    presentPaternToSOM(data);
    return &activationMap;
}


///Persistence functions

std::ostream& TSOM::toStream(std::ostream& out)
{
   out <<  "somRows" << CFGFile::delimiterStr << neurons.rows() << CFGFile::endValueStr << endl;
   out <<  "somCols" <<  CFGFile::delimiterStr << neurons.cols() << CFGFile::endValueStr << endl;
   out <<  "inputRows" << CFGFile::delimiterStr << inputRows << CFGFile::endValueStr << endl;
   out <<  "inputCols" << CFGFile::delimiterStr << inputCols << CFGFile::endValueStr << endl;

   for (uint r=0; r<neurons.rows(); r++)
   {
        for (uint c=0; c<neurons.cols(); c++)
           out << neurons[r][c] << endl;
   }

   return out;
}

std::istream& TSOM::fromStream(std::istream& in)
{
    uint somRows, somCols;
	string line;

	while (getline(in, line))
	{
		if (checkEndSection(line))
		  break;

        if (getValue(line, somRows))
        {
            getline(in, line);
            getValue(line, somCols);

            getline(in, line);
            getValue(line, inputRows);

            getline(in, line);
            getValue(line, inputCols);

            initialize(somRows, somCols, inputRows, inputCols);
            for (uint r=0; r<neurons.rows(); r++)
            {
                for (uint c=0; c<neurons.cols(); c++)
                   in >> neurons[r][c];
            }
		}
	}

   return in;
}

MatMatrix<TNeuron> *TSOM::getNeurons()
{
  return &neurons;
}

/// Class TNeuron ///
TNeuron::TNeuron()
{
    set(0,0);
    wl = wr = 0;
    maxActivation = 0;
}

TNeuron::TNeuron(uint r, uint c)
{
    set(r,c);
}

void TNeuron::setupWeights(uint wr, uint wc)
{
    weights.size(wr, wc);

    weights.random();
    weights = weights.mult(3);
    weights = weights.add(-1.5);
}

void TNeuron::set(uint r, uint c)
{
   this->r = r;
   this->c = c;
}

std::ostream& operator << (std::ostream& ost, TNeuron &neuron)
{
   ost << "rows" << CFGFile::delimiterStr << neuron.weights.rows() << CFGFile::endValueStr << endl;
   ost << "cols" << CFGFile::delimiterStr << neuron.weights.cols() << CFGFile::endValueStr << endl;

   ost << "weights" << CFGFile::delimiterStr << "\t";
   for (uint r=0; r<neuron.weights.rows(); r++)
   {
        for (uint c=0; c<neuron.weights.cols(); c++)
        {
           ost << neuron.weights[r][c] << "\t";
        }
   }

   return ost;
}

std::istream& operator >> (std::istream& ist, TNeuron &neuron)
{
   uint rows, cols;
   std::string line;

   getline(ist, line);
   getValue(line, rows);

   getline(ist, line);
   getValue(line, cols);

   neuron.weights.size(rows, cols);
   ist >> line; //read "weights:";

   for (uint r=0; r<neuron.weights.rows(); r++)
   {
        for (uint c=0; c<neuron.weights.cols(); c++)
           ist >> neuron.weights[r][c];
   }

   getline(ist, line); //read '\t';

   return ist;
}


TNeuron::~TNeuron()
{

}
