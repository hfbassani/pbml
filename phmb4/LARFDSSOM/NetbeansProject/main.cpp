/*
 * File:   main.cpp
 * Author: hans
 *
 * Created on 11 de Outubro de 2010, 07:25
 */

#include <stdlib.h>
#include <fstream>
#include <algorithm>
#include <time.h>
#include <limits>
#include <unistd.h>
#include "Defines.h"
#include "DebugOut.h"
#include "ClusteringMetrics.h"
#include "ArffData.h"
#include "ClusteringSOM.h"
#include "LARFDSSOM.h"
#include "DisplayMap.h"
#include <sys/stat.h>

class DisplaySSHSOM: public DisplayMap {

    LARFDSSOM *som;

public:
    
    DisplaySSHSOM(LARFDSSOM *som, MatMatrix<float> *trainingData, MatMatrix<float> *averages = NULL, map<int, int> *groupLabels = NULL, int padding = 20, int gitter = 0, bool bmucolor = true, bool trueClustersColor = true, bool filterNoise = false):DisplayMap(trainingData, averages, groupLabels, padding, gitter, bmucolor, trueClustersColor, filterNoise) {
        this->som = som;
    }

    virtual void plotMap(CImg<unsigned char> *image, bool drawNodes, bool drawConections) {
        
        som->enumerateNodes();
        unsigned char bmuColor[3];
        int width = (image->width() - 2 * padding);
        int height = (image->height() - 2 * padding);

        if (drawNodes) {
            MatMatrix<float> centers;
            som->outputCenters(centers);
            for (DSNode *bmu = som->getFirstNode(); !som->finished(); bmu = som->getNextNode()) {
                int r, g, b;
                int size = som->size()-1;
                if (size==0) size = 1;
                int h = HUE_START + bmu->getId()*MAX_HUE / (size);
                HSVtoRGB(&r, &g, &b, h, 255, 255);
                bmuColor[0] = r;
                bmuColor[1] = g;
                bmuColor[2] = b;

                image->draw_circle(padding + bmu->w[X] * width, padding + bmu->w[Y] * height, 4, contour);
                image->draw_circle(padding + bmu->w[X] * width, padding + bmu->w[Y] * height, 3, bmuColor);
                int cx = bmu->w[X] * width;
                int cy = bmu->w[Y] * height;
                int x0 = bmu->w[X] * width - bmu->ds[X]*20;
                int x1 = bmu->w[X] * width + bmu->ds[X]*20;
                int y0 = bmu->w[Y] * height - bmu->ds[Y]*20;
                int y1 = bmu->w[Y] * height + bmu->ds[Y]*20;
                image->draw_line(padding + cx, padding + y0, padding + cx, padding + y1, contour);
                image->draw_line(padding + x0, padding + cy, padding + x1, padding + cy, contour);
            }
            
            //Draw connections
            if (drawConnections) {
                LARFDSSOM::TPConnectionSet::iterator it;
                for (it = som->meshConnectionSet.begin(); it != som->meshConnectionSet.end(); it++) {
                    float x0 = (*it)->node[0]->w[X];
                    float y0 = (*it)->node[0]->w[Y];
                    float x1 = (*it)->node[1]->w[X];
                    float y1 = (*it)->node[1]->w[Y];
                    float dist = (*it)->node[0]->ds.dist((*it)->node[1]->ds) / sqrt((*it)->node[1]->ds.size());
                    unsigned char conColor[] = {255, 255, 255};
                    conColor[0] = conColor[0] * dist;
                    conColor[1] = conColor[1] * dist;
                    conColor[2] = conColor[2] * dist;
                    image->draw_line(padding + x0 * width, padding + y0 * height, padding + x1 * width, padding + y1 * height, conColor);
                }
            }
        }
    }
};

void runExperiments (std::vector<float> params, string filePath, string outputPath, float supervisionRate,
        bool isSubspaceClustering, bool isFilterNoise, bool sorted);
void runTestTrainExperiments (std::vector<float> params, string filePath, string testPath, string outputPath, float supervisionRate,
        bool isSubspaceClustering, bool isFilterNoise, bool sorted);
void evaluate (string filePath, string somPath, bool subspaceClustering, bool filterNoise, bool sorted);
std::vector<float> loadParametersFile(string path);
std::vector<string> loadStringFile(string path);
string getFileName(string filePath);
int findLast(const string str, string delim);

int main(int argc, char** argv) {

    dbgThreshold(1);

    dbgOut(1) << "Running LARFDSSOM" << endl;

    string inputPath = "";
    string testPath = "";
    string resultPath = "";
    string mapPath = "";
    string parametersFile = "";

    bool isSubspaceClustering = true;
    bool isFilterNoise = true;
    bool isSorted = false;
    
    bool runTrainTest = false;
    bool runEvaluation = false;
    
    float supervisionRate = -1;
    
    int c;
    while ((c = getopt(argc, argv, "i:t:r:p:l:m:sfSce")) != -1) {

        switch (c) {
            case 'i':
                inputPath.assign(optarg);
                break;
            case 't':
                testPath.assign(optarg);
                break;
            case 'r':
                resultPath.assign(optarg);
                break;
            case 'p':
                parametersFile.assign(optarg);
                break;
            case 'l':
                supervisionRate = atof(optarg);
                break;
            case 'm':
                mapPath.assign(optarg);
                break;
            case 's':
                isSubspaceClustering = false;
                break;
            case 'f':
                isFilterNoise = false;
                break;
            case 'S':
                isSorted = true;
                break;
            case 'c':
                runTrainTest = true;
                break;
            case 'e':
                runEvaluation = true;
                break;
        }
    }

    
    if (runEvaluation) {
        evaluate(inputPath, mapPath, isSubspaceClustering, isFilterNoise, isSorted);
        return 0;
    }
    
    if (mkdir(resultPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0) {
        dbgOut(1) << "Results Directory Created" << endl;
    } 
    
    std::vector<string> inputFiles = loadStringFile(inputPath);
    std::vector<string> testFiles = loadStringFile(testPath);
    std::vector<float> params = loadParametersFile(parametersFile);

    for (int i = 0 ; i < inputFiles.size() - 1 ; ++i) {
        if(!runTrainTest) {
            runExperiments(params, inputFiles[i], resultPath, supervisionRate, isSubspaceClustering, isFilterNoise, isSorted);
        } else {
            runTestTrainExperiments(params, inputFiles[i], testFiles[i], resultPath, supervisionRate, isSubspaceClustering, isFilterNoise, isSorted);
        }
    }
}

void evaluate (string filePath, string somPath, bool subspaceClustering, bool filterNoise, bool sorted) {
    LARFDSSOM som(1);
    SOM<DSNode> *dssom = (SOM<DSNode>*) &som;
    som.noCls = -1;
    som.readSOM("/home/pedro/Documents/git/pbml/phmb4/LARFDSSOM/NetbeansProject/test_folds_pls_jesus_breast2/som_train_breast_x2_k2_282");
    
    ClusteringMeshSOM clusteringSOM(dssom);
    clusteringSOM.readFile("/home/pedro/Documents/git/pbml/Datasets/Realdata_3Times3Folds_TestBreast/test_breast_x2_k2.arff");
    clusteringSOM.sorted = sorted;
    
    
    clusteringSOM.setIsSubspaceClustering(subspaceClustering);
    clusteringSOM.setFilterNoise(filterNoise);
    
//    clusteringSOM.writeClusterResults("/home/pedro/Documents/teste.results");
    clusteringSOM.outConfusionMatrix(clusteringSOM.groups, clusteringSOM.groupLabels);
    clusteringSOM.outClusters();
}

void runExperiments (std::vector<float> params, string filePath, string outputPath, float supervisionRate,
        bool isSubspaceClustering, bool isFilterNoise, bool sorted) {

    LARFDSSOM som(1);
    SOM<DSNode> *dssom = (SOM<DSNode>*) &som;

    ClusteringMeshSOM clusteringSOM(dssom);
    clusteringSOM.readFile(filePath);
    clusteringSOM.sorted = sorted;

    clusteringSOM.setIsSubspaceClustering(isSubspaceClustering);
    clusteringSOM.setFilterNoise(isFilterNoise);    
    
    int numberOfParameters = 12;
    
    for (int i = 0 ; i < params.size() - 1 ; i += numberOfParameters) {
        som.a_t = params[i];
        som.lp = params[i + 1];
        som.dsbeta = params[i + 2];
        som.age_wins = params[i + 3];
        som.e_b = params[i + 4];
        som.e_n = params[i + 5] * som.e_b;
        som.epsilon_ds = params[i + 6];
        som.minwd = params[i + 7];
        som.epochs = params[i + 8];
        som.push_rate = params[i + 9] * som.e_b;
        
        if (supervisionRate < 0) 
            som.supervisionRate = params[i + 10];
        else 
            som.supervisionRate = supervisionRate;
        
        string index = std::to_string((i/numberOfParameters));
        
        som.unsupervisionRate = 1.0 - som.supervisionRate;
                  
        srand(params[i + 11] + time(NULL));
        som.noCls = std::min_element(clusteringSOM.groups.begin(), clusteringSOM.groups.end())[0] - 1;
        som.maxNodeNumber = 140;
        som.age_wins = round(som.age_wins*clusteringSOM.getNumSamples());
        som.reset(clusteringSOM.getInputSize());
        clusteringSOM.trainSOM(som.epochs);
        som.finishMapFixed(sorted, clusteringSOM.groups);
        clusteringSOM.writeClusterResults(outputPath + getFileName(filePath) + "_" + index + ".results");
    }
}

void runTestTrainExperiments (std::vector<float> params, string filePath, string testPath, string outputPath, float supervisionRate,
        bool isSubspaceClustering, bool isFilterNoise, bool sorted) { 
    
    int numberOfParameters = 12;
    
    for (int i = 0 ; i < params.size() - 1 ; i += numberOfParameters) {
        LARFDSSOM som(1);
        SOM<DSNode> *dssom = (SOM<DSNode>*) &som;

        ClusteringMeshSOM clusteringSOM(dssom);
        clusteringSOM.readFile(filePath);
        clusteringSOM.sorted = sorted;

        clusteringSOM.setIsSubspaceClustering(isSubspaceClustering);
        clusteringSOM.setFilterNoise(isFilterNoise);   
    							
//        som.a_t = 0.850051;//params[i];
//        som.lp = 0.00683292;//params[i + 1];
//        som.dsbeta = 0.135774;//params[i + 2];
//        som.age_wins = 63;//params[i + 3];
//        som.e_b = 0.00657346;//params[i + 4];
//        som.e_n = 0.0288698 * som.e_b;
//        som.epsilon_ds = 0.0237034;params[i + 6];
//        som.minwd = 0.324148;//0;//params[i + 7]; 
//        som.epochs = 19;//params[i + 8];
//        som.push_rate = 0.132613 * som.e_b;
        
//        som.a_t = 0.893248;//params[i];
//        som.lp = 0.00693669;//params[i + 1];
//        som.dsbeta = 0.274298;//params[i + 2];
//        som.age_wins = 49;//params[i + 3];
//        som.e_b = 0.0591604;//params[i + 4];
//        som.e_n = 0.474905 * som.e_b;
//        som.epsilon_ds = 0.0482915;params[i + 6];
//        som.minwd = 0.0571491;//0;//params[i + 7]; 
//        som.epochs = 96;//params[i + 8];
//        som.push_rate = 0.422377 * som.e_b;
                
        som.a_t = params[i];
        som.lp = params[i + 1];
        som.dsbeta = params[i + 2];
        som.age_wins = params[i + 3];
        som.e_b = params[i + 4];
        som.e_n = params[i + 5] * som.e_b;
        som.epsilon_ds = params[i + 6];
        som.minwd = params[i + 7]; 
        som.epochs = params[i + 8];
        som.push_rate = params[i +9] * som.e_b;
        
        if (supervisionRate < 0) 
            som.supervisionRate = params[i + 10];//0.823789
        else 
            som.supervisionRate = supervisionRate;
        
        string index = std::to_string((i/numberOfParameters));
        
        som.unsupervisionRate = 1.0 - som.supervisionRate;

//        srand(21.2613 + time(NULL));
//        srand(59.7078 + i);
        srand(params[i + 11]);

        som.noCls = std::min_element(clusteringSOM.groups.begin(), clusteringSOM.groups.end())[0] - 1;
        som.maxNodeNumber = clusteringSOM.trainingData->size();
        som.age_wins = round(som.age_wins*clusteringSOM.getNumSamples());
        som.reset(clusteringSOM.getInputSize());
        //clusteringSOM.trainSOM(som.epochs);
        
        DisplaySSHSOM dm(&som, clusteringSOM.trainingData);
        
        for (int epoch = 0; epoch < som.epochs; epoch++)
            for (int row = 0 ; row < clusteringSOM.trainingData->rows() ; ++row) {
                som.trainningStep(rand()%clusteringSOM.trainingData->rows(), clusteringSOM.groups);
                dm.display();
            }
        
        som.finishMapFixed(sorted, clusteringSOM.groups);
        som.saveSOM(outputPath + "som_" + getFileName(filePath) + "_" + index);
        
        dm.displayLoop();
        dm.close();

//        som.checkNodeClasses(outputPath + getFileName(testPath) + "_" + index + ".results.noclass");
        clusteringSOM.cleanUpTrainingData();
        clusteringSOM.readFile(testPath);
        clusteringSOM.writeClusterResults(outputPath + getFileName(testPath) + "_" + index + ".results");

//        som.enumerateNodes();
//        dbgOut(1) << clusteringSOM.outClassInfo(clusteringSOM.groups, clusteringSOM.groupLabels) << endl;
//        clusteringSOM.outConfusionMatrix(clusteringSOM.groups, clusteringSOM.groupLabels);
    }
}

std::vector<float> loadParametersFile(string path) {
    std::ifstream file(path.c_str());
    std::string text;
    std::vector<float> params;
    while (!file.eof()) {
        getline(file, text);
        params.push_back(std::atof(text.c_str()));
    }
    return params;
}

std::vector<string> loadStringFile(string path) {
    std::ifstream file(path.c_str());
    std::string text;
    std::vector<string> params;
    while (!file.eof()) {
        getline(file, text);
        params.push_back(text.c_str());
    }
    return params;
}

string getFileName(string filePath) {

    int start = findLast(filePath, "/");
    int end = findLast(filePath, ".");

    return filePath.substr(start + 1, end - start - 1);
}

int findLast(string str, string delim) {
    std::vector<int> splits;

    int current, previous = 0;
    current = str.find(delim);
    while (current != std::string::npos) {
        splits.push_back(current);
        previous = current + 1;
        current = str.find(delim, previous);
    }

    return splits[splits.size() - 1];
}
