/* 
 * File:   SubspaceClusteringSOM.h
 * Author: hans
 *
 * Created on 4 de Outubro de 2011, 16:49
 */

#ifndef SUBSPACECLUSTERINGSOM_H
#define	SUBSPACECLUSTERINGSOM_H

#include <map>
#include "Parameters.h"
#include "MatMatrix.h"
#include "DSSOM.h"
#include "MWDSSOM.h"
#include "ArffData.h"
#include "ClusteringMetrics.h"
#include "ClusteringSOM.h"

class SCSOMParameters : public Parameters {
public:

    SCSOMParameters() {

        //Comentários do início do arquivo
        comments = "SCSOMParameters Parameters";

        //Definição do nome da sessão do arquivo na qual estes parâmetros serão salvos
        section = "SCSOMParameters";
    }
};

template<class TypeSOM>
class SubspaceClusteringSOM {
    MatMatrix<float> trainingData;
    std::vector<int> groups;
    std::map<int, int> groupLabels;
    int group;

    ClusteringSOM<TypeSOM> *som;

public:
    SCSOMParameters parameters;

    SubspaceClusteringSOM(ClusteringSOM<TypeSOM> *som) {
        this->som = som;
    }
    
    void cleanUpTrainingData() {
        trainingData.clear();
        groups.clear();
        groupLabels.clear();
    }

    bool readFile(const std::string filename) {

        cleanUpTrainingData();
        if (ArffData::readArff(filename, trainingData, groupLabels, groups)) {
                ArffData::rescaleCols01(trainingData);
                return true;
        }
        return false;
    }

    void trainSOM(const DSSOMParameters& somParameters) {
        //train som
        som->train(trainingData);
    }

    bool writeClusterResults(const std::string filename) {

        ofstream file;
        file.open(filename.c_str());

        if (!file.is_open()) {
            dbgOut(0) << "Error openning output file" << endl;
            return false;
        }

        int meshSize = som->getMeshSize();
        int inputSize = som->getInputSize();

        file << meshSize << "\t" << inputSize << endl;

        for (int i = 0; i < meshSize; i++) {
            MatVector<float> relevances;
            som->getRelevances(i, relevances);
            
            for (int j = 0; j < inputSize; j++) {
                file << relevances[j];
                if (j != inputSize - 1)
                    file << "\t";
            }
            file << endl;
        }

        for (int i = 0; i < trainingData.size(); i++) {
            MatVector<float> sample;
            trainingData.getRow(i, sample);
            std::vector<int> winners;
            som->getWinners(sample, winners);

            file << i << "\t";
            for (int j = 0; j < winners.size(); j++) {
                file << winners[j];
                if (j != winners.size() - 1)
                    file << "\t";
            }
            file << endl;
        }

        file.close();
        return true;
    }
};



/*Paper
    DSNeuron bmu;
    for (int i=0;i<trainingData.rows(); i++) {
        MatVector<float> sample;
        trainingData.getRow(i, sample);

        som.resetRelevances();
        int k=1;
        while (som.getRelevance().max()>som.parameters.epsilonRho && k<=som.parameters.numWinners)
        {
           if (k==1) {
               float activation = som.findFirstBMU(sample, bmu);
               if (activation < som.parameters.outliersThreshold) {
                   file << i << "\t" << -1 << endl;
                   break;
               }
           }
           else {
               som.findNextBMU(sample, bmu);
           }
           file << i << "\t" << (bmu.r + bmu.c*som.parameters.NY) << endl;           
           som.updateRelevances(bmu);
           k = k + 1;
        }
    }
    /**/

#endif	/* SUBSPACECLUSTERINGSOM_H */