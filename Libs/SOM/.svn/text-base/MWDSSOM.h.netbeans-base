/* 
 * File:   FSOM.h
 * Author: hans
 *
 * Created on 6 de Outubro de 2010, 17:41
 */

#ifndef _MWDSSOM_H
#define	_MWDSSOM_H

#include "DSSOM.h"
#include "DSNeuron.h"

class MWDSSOMParameters: public DSSOMParameters {

public:

    Parameter<uint> numWinners;  //Number of winners per pattern

	MWDSSOMParameters():DSSOMParameters() {

		//Comentários do início do arquivo
		comments = "Multiple Winners Dimension Selective Self-Organizing Map Parameters";

		//Definição do nome da sessão do arquivo na qual estes parâmetros serão salvos
		section = "MWDSSOMParameters";

		//Parâmetros persistentes
		addParameterD(numWinners, "Number of winners per pattern");

		//Default values
                numWinners      = 1;    //Number of winners per pattern
	}
};

class MWDSSOM: public DSSOM<DSNeuron> {

private:
    MatVector<float> relevance; // Relevance of current input information features

public:
    //MWDSSOMParameters parameters;
    
    MWDSSOM();
    virtual void initialize(uint somRows, uint somCols, uint inpuSize);
    virtual float findFirstBMU(const MatVector<float> &data, DSNeuron &bmu);
    virtual float findNextBMU(const MatVector<float> &data, DSNeuron &bmu);
    virtual float findNextBMU(DSNeuron &bmu);
    virtual void train(MatMatrix<float> &trainingData);
    virtual void trainStep(MatMatrix<float> &trainingData);
    virtual void deactivateNeighbors(DSNeuron &neuron);
    virtual float presentPaternToNeuronAgain(const MatVector<float> &data, DSNeuron &neuron);
    virtual float presentPaternToNeuron(const MatVector<float> &data, DSNeuron &neuron);
    virtual void resetRelevances();
    virtual void updateRelevances(DSNeuron &bmu);
    const MatVector<float> getRelevance();

};


#endif	/* _MWDSSOM_H */

