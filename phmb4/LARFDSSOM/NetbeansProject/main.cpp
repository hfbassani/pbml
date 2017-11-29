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
#include "MatMatrix.h"
#include "MatVector.h"
#include "Defines.h"
#include "DebugOut.h"
#include "ClusteringMetrics.h"
#include "ArffData.h"
#include "ClusteringSOM.h"
#include "LARFDSSOM.h"

using namespace std;

void runExperiments (std::vector<float> params, string filePath, string outputPath,
        bool isSubspaceClustering, bool isFilterNoise, bool sorted);
std::vector<float> loadParametersFile(string path);
std::vector<string> loadStringFile(string path);
int findLast(const string str, string delim);
string getFileName(string filePath);


int main(int argc, char** argv) {

    dbgThreshold(1);

    dbgOut(1) << "Running LARFDSSOM" << endl;

    string inputPath = "";
    string resultPath = "";
    string parametersFile = "";

    bool isSubspaceClustering = true;
    bool isFilterNoise = true;
    bool isSorted = false;

    int c;
    while ((c = getopt(argc, argv, "i:r:p:sfS")) != -1) {
        switch (c) {
            case 'i':
                inputPath.assign(optarg);
                break;
            case 'r':
                resultPath.assign(optarg);
                break;
            case 'p':
                parametersFile.assign(optarg);
                break;
            case 's':
                isSubspaceClustering = false;
                break;
            case 'f':
                isFilterNoise = false;
                break;
            case 'S':
                //TODO: merge from branch hybrid-tests
//                isSorted = true;
                break;
        }
    }

    std::vector<string> inputFiles = loadStringFile(inputPath);
    std::vector<float> params = loadParametersFile(parametersFile);

    for (int i = 0 ; i < inputFiles.size() - 1 ; ++i) {
        runExperiments(params, inputFiles[i], resultPath, isSubspaceClustering, isFilterNoise, isSorted);
    }
}

void runExperiments (std::vector<float> params, string filePath, string outputPath,
        bool isSubspaceClustering, bool isFilterNoise, bool sorted) {

    LARFDSSOM som(1);
    SOM<DSNode> *dssom = (SOM<DSNode>*) &som;

    ClusteringMeshSOM clusteringSOM(dssom);
    clusteringSOM.readFile(filePath);
    clusteringSOM.sorted = sorted;

    clusteringSOM.setIsSubspaceClustering(isSubspaceClustering);
    clusteringSOM.setFilterNoise(isFilterNoise);

    int numberOfParameters = 10;

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

        string index = std::to_string((i/numberOfParameters));

        srand(params[i + 9] + time(NULL));
        som.maxNodeNumber = 70;
        som.age_wins = round(som.age_wins*clusteringSOM.getNumSamples());
        som.reset(clusteringSOM.getInputSize());
        clusteringSOM.trainSOM(som.epochs);
        som.finishMapFixed(sorted);

        clusteringSOM.writeClusterResults(outputPath + getFileName(filePath) + "_" + index + ".results");

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
