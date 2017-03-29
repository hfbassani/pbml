
#ifndef SOM2D_H
#define SOM2D_H

#include "MatMatrix.h"
#include "Parameters.h"
#include "Neuron.h"
#include <vector>

class SOMParameters: public Parameters {

public:

    Parameter<uint> NX; //Map width
    Parameter<uint> NY; //Map height
    Parameter<uint> NFeatures; //Number of input data columns
    Parameter<uint> numNeighbors; //num neuron neigbors
    Parameter<float> alpha;       //learning rate
    Parameter<float> lambda2;     //learning rate decay (controls tau2)
    Parameter<float> sigma;       //neighborhood function standard deviation
    Parameter<float> lambda1;     //neighborhood function decay (controls tau1)
    Parameter<uint> tmax;         //number of iterations

	SOMParameters() {

		//Comentários do início do arquivo
		comments = "TSOM Parameters";

		//Definição do nome da sessão do arquivo na qual estes parâmetros serão salvos
		section = "TSOMParameters";

		//Parâmetros persistentes
                addParameterD(NX, "Map width");
                addParameterD(NY, "Map height");
                addParameterD(NFeatures, "Number of input data columns");
		addParameterD(numNeighbors, "City block distance to the farthest neigbors");
		addParameterD(alpha, "Learning rate");
		addParameterD(lambda2, "Learning rate decay with time (defines tau2 as % tmax)");
		addParameterD(sigma, "Neighborhood function standard deviation");
                addParameterD(lambda1, "neighborhood function decay (defines tau1 as % tmax)");
		addParameterD(tmax, "Number of training iterations");

		//Default values                
                NX = 4;
                NY = 4;
                NFeatures = 3;
                numNeighbors = 1;
                lambda2 = 1;
                alpha = 0.01;
                sigma = 0.5;
                lambda1 = 1;
                tmax = 1;
	}
};

template <class TNeuron = Neuron, class TSOMParameters = SOMParameters>
class SOM2D: public CFGFileObject
{
protected:
    MatMatrix<TNeuron> neurons;
    MatMatrix<float> activationMap;
    uint somRows;
    uint somCols;
    uint inputSize;
    uint t;

    float alpha_t;
    float sigma_t;

    public:
        TSOMParameters parameters;

    public:
        SOM2D();
        virtual ~SOM2D();
        bool (*callback)(SOM2D<TNeuron, TSOMParameters> &som);
        virtual void initialize(uint somRows, uint somCols, uint inpuSize);
        virtual void initializeP(const TSOMParameters &parameters);
        virtual void reRandomizeWeights();
        virtual void setupGridWeights(MatVector<float> &center, MatVector<float> &delta);
        virtual void setParameters(uint numNeighbors, float alpha, float lambda, float sigma, uint tmax);
        
        virtual void train(MatMatrix<float> &trainingData);
        virtual void trainStep(MatMatrix<float> &trainingData);
        virtual float findBMU(const MatVector<float> &data, TNeuron &bmu);

        virtual void updateActivationMap(MatVector<float> &data);
        virtual void getActivationMap(MatMatrix<float> &activationMap);
        virtual void getClusterPosition(const MatVector<float> &data, uint &r, uint &c);
        virtual void getNeuron(TNeuron &neuron);
        virtual void getNeurons(MatMatrix<TNeuron> &neurons) const;
        virtual void getNeuronWeights(TNeuron &neuron, MatVector<float> & neuronWeights);

        virtual bool incTime();
        virtual void resetTime();
        float getAlpha_t() const;
        float getSigma_t() const;
        uint getT() const;
        uint getInputSize() const;
        uint getSomCols() const;
        uint getSomRows() const;
        TSOMParameters getParameters() const; 
    
        virtual bool saveParameters(std::ofstream &file) {

            file << parameters.NX << "\t";
            file << parameters.NY << "\t";
            file << parameters.NFeatures<< "\t";
            file << parameters.numNeighbors<< "\t";
            file << parameters.lambda2<< "\t";
            file << parameters.alpha<< "\t";
            file << parameters.sigma<< "\t";
            file << parameters.lambda1<< "\t";
            file << parameters.tmax<< "\n";
            return true;
        }

        virtual bool saveSOM(const std::string &filename) {

            std::ofstream file(filename.c_str());
            file.precision(16);

            if (!file.is_open()) {
                return false;
            }

            saveParameters(file);

            for (int c=0; c<getSomCols(); c++)
            for (int r=0; r<getSomRows(); r++) {
                neurons[r][c].write(file);
            }

            file.close();
            return true;
        }

        virtual bool readParameters(std::ifstream &file) {

            file >> parameters.NX.value;
            file >> parameters.NY.value;
            file >> parameters.NFeatures.value;
            file >> parameters.numNeighbors.value;
            file >> parameters.lambda2.value;
            file >> parameters.alpha.value;
            file >> parameters.sigma.value;
            file >> parameters.lambda1.value;
            file >> parameters.tmax.value;
            file.get();//skip line end
            return true;
        }

        virtual bool readSOM(const std::string &filename) {

            std::ifstream file(filename.c_str());

            if (!file.is_open()) {
                return false;
            }

            readParameters(file);
            
            initialize(parameters.NX,parameters.NY, parameters.NFeatures);

            for (int c=0; c<getSomCols(); c++)
            for (int r=0; r<getSomRows(); r++) {
                neurons[r][c].read(file);
            }

            return true;
        }        
    protected:
        virtual float presentPaternToNeuron(const MatVector<float> &data, TNeuron &neuron);
        virtual void updateNeuron(TNeuron &neuron, MatVector<float> &data, float alpha, float phi);
        virtual void presentPaternToSOM(const MatVector<float> &data);
        virtual void updateNeigbors(TNeuron &neuron, MatVector<float> &data, float alpha);
        float phi(float dr, float dc);

        std::ostream& toStream(std::ostream& out);
        std::istream& fromStream(std::istream& in);
};

#include "SOM2D.cpp"

#endif // SOM2D_H
