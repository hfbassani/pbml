/*
 *  Created on: 2017
 *      Author: phmb4
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
#include "WIP.h"
#include "WIPNode.h"
#include <sys/stat.h>

void runTrainTestExperiments (std::vector<float> params, string filePath, string testPath, string outputPath, 
        bool isSubspaceClustering, bool isFilterNoise, bool sorted, bool normalize, bool keepMapSaved, 
        bool removeNodes, int numNodes, bool runTrainTest);
void evaluate (string filePath, string somPath, bool subspaceClustering, bool filterNoise, bool sorted, bool normalize);

std::vector<float> loadParametersFile(string path);
std::vector<string> loadStringFile(string path);
string getFileName(string filePath);
int findLast(const string str, string delim);

int main(int argc, char** argv) {

    dbgThreshold(0);

    dbgOut(0) << "Running WIP" << endl;

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
    
    bool normalize = false;
    
    bool keepMapSaved = false;
    
    bool removeNodes = true;
    
    int numNodes = 250;
    
    int c;
    while ((c = getopt(argc, argv, "i:t:r:p:m:N:sfScednkz")) != -1) {

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
            case 'm':
                mapPath.assign(optarg);
                break;
            case 'N':
                numNodes = atoi(optarg);
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
            case 'n':
                normalize = true;
                break;
            case 'k':
                keepMapSaved = true;
                break;
            case 'z':
                removeNodes = false;
                break;
        }
    }

    
    if (runEvaluation) {
        evaluate(inputPath, mapPath, isSubspaceClustering, isFilterNoise, isSorted, normalize);
        return 0;
    }
    
    if (mkdir(resultPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0) {
        dbgOut(0) << "Results Directory Created" << endl;
    } 
    
    std::vector<string> inputFiles = loadStringFile(inputPath);
    
    std::vector<string> testFiles = inputFiles;
    if (testPath != "") {
        testFiles = loadStringFile(testPath);
    }
    std::vector<float> params = loadParametersFile(parametersFile);

    for (int i = 0 ; i < inputFiles.size() - 1 ; ++i) {
        runTrainTestExperiments(params, inputFiles[i], testFiles[i], resultPath, isSubspaceClustering, 
                isFilterNoise, isSorted, normalize, keepMapSaved, removeNodes, numNodes, runTrainTest);
    }
}

void evaluate (string filePath, string somPath, bool subspaceClustering, bool filterNoise, bool sorted, bool normalize) {
    WIP som(1);
    SOM<WIPNode> *sssom = (SOM<WIPNode>*) &som;
    som.noCls = 999;
    som.readSOM(somPath);
    
    ClusteringMeshWIP clusteringSOM(sssom);
    clusteringSOM.readFile(filePath, normalize);
    clusteringSOM.sorted = sorted;
    
    
    clusteringSOM.setIsSubspaceClustering(subspaceClustering);
    clusteringSOM.setFilterNoise(filterNoise);
    
    clusteringSOM.outConfusionMatrix(clusteringSOM.groups, clusteringSOM.groupLabels);
    clusteringSOM.outClassInfo(clusteringSOM.groups, clusteringSOM.groupLabels);
}

void runTrainTestExperiments (std::vector<float> params, string filePath, string testPath, string outputPath, 
        bool isSubspaceClustering, bool isFilterNoise, bool sorted, bool normalize, bool keepMapSaved, 
        bool removeNodes, int numNodes, bool runTrainTest) { 
    
    int numberOfParameters = 9;
    
    for (int i = 0 ; i < params.size() - 1 ; i += numberOfParameters) {
        WIP som(1);
        SOM<WIPNode> *sssom = (SOM<WIPNode>*) &som;
        
        som.unsup_win = 0;
        som.unsup_create = 0;
        som.unsup_else = 0;
        som.sup_win = 0;
        som.sup_create = 0;
        som.sup_else = 0;
        som.sup_handle_new_win_full = 0;
        som.sup_handle_new_win_relevances = 0;
        som.sup_handle_create = 0;
        som.sup_handle_else = 0;

        ClusteringMeshWIP clusteringSOM(sssom);
        clusteringSOM.readFile(filePath, normalize);
        clusteringSOM.sorted = sorted;

        clusteringSOM.setIsSubspaceClustering(isSubspaceClustering);
        clusteringSOM.setFilterNoise(isFilterNoise);   
           
        if (removeNodes) {
            som.lp = params[i];//pow(10, params[i]);
        } else {
            som.lp = 0.0;            
        }
        som.dsbeta = params[i + 1];
        som.age_wins = params[i + 2];
        som.e_b = params[i + 3];
        som.e_n = params[i + 4] * som.e_b;
        som.epsilon_ds = params[i + 5];
        som.minwd = params[i + 6]; 
        som.epochs = params[i + 7];
        srand(params[i + 8]);
        
        string index = std::to_string((i/numberOfParameters));
                  
        som.noCls = 999;
        som.maxNodeNumber = numNodes;
        som.age_wins = round(som.age_wins*clusteringSOM.getNumSamples());
        som.reset(clusteringSOM.getInputSize());
        
        clusteringSOM.trainSOM(som.epochs);
        
        som.finishMapFixed(sorted, clusteringSOM.groups, clusteringSOM.groupLabels);
        
        if (keepMapSaved) {
            som.saveSOM(outputPath + "som_" + getFileName(filePath) + "_" + index);
        }

        
        if (runTrainTest) {
            clusteringSOM.cleanUpTrainingData();
            clusteringSOM.readFile(testPath, normalize);
            clusteringSOM.writeClusterResults(outputPath + getFileName(testPath) + "_" + index + ".results");
        } else {
            clusteringSOM.writeClusterResults(outputPath + getFileName(filePath) + "_" + index + ".results");
        }

                
//        dbgOut(1) << "-----------------------------" << endl;
//        dbgOut(1) << "som.unsup_win:" << som.unsup_win << endl;
//        dbgOut(1) << "som.unsup_create:" << som.unsup_create << endl;
//        dbgOut(1) << "som.unsup_else:" << som.unsup_else << endl;
//        dbgOut(1) << "som.sup_win:" << som.sup_win << endl;
//        dbgOut(1) << "som.sup_create:" << som.sup_create << endl;
//        dbgOut(1) << "som.sup_else:" << som.sup_else << endl;
//        dbgOut(1) << "som.sup_handle_new_win_full:" << som.sup_handle_new_win_full << endl;
//        dbgOut(1) << "som.sup_handle_new_win_relevances:" << som.sup_handle_new_win_relevances << endl;
//        dbgOut(1) << "som.sup_handle_create:" << som.sup_handle_create << endl;
//        dbgOut(1) << "som.sup_handle_else:" << som.sup_handle_else << endl;
        
//        clusteringSOM.outAccuracy(clusteringSOM.groups, clusteringSOM.groupLabels);
        
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
