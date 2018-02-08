/* 
 * File:   main.cpp
 * Author: flavia
 *
 * Created on 27 de Junho de 2014, 07:25
 */

#include <iomanip>

#include "LatinHypercubeSampling.h"
#include "GRLVQ.h"
#include "ArffData.h"

#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <sys/stat.h>

#include "MatUtils.h"
#include "MatVector.h"
#include "MatMatrix.h"

using namespace std;

std::vector<float> loadParametersFile(string path);
std::vector<string> loadStringFile(string path);
bool readFile(const std::string &filename, MatMatrix<float> &data, vector<int> &classes, map<int,int> &labels);
string getFileName(string filePath);
int findLast(string str, string delim);

void runLHS();
void runExperiments(std::vector<float> params, string filePath, string testPath, string outputPath);
bool trainLVQ(GRLVQ &lvq, int nNodes, int epochs, unsigned int seed, MatMatrix<float> &trainingData, vector<int> &classes, map<int,int> &labels);
void classifyTestData (string testPath, string outputPath, GRLVQ &lvq);

int main(int argc, char** argv) {

    dbgThreshold(1);

    dbgOut(1) << "Running GRLVQ" << endl;

    string inputPath = "";
    string testPath = "";
    string resultPath = "";
    string parametersFile = "";
        
    int c;
    while ((c = getopt(argc, argv, "i:t:r:p:")) != -1) {

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
        }
    }
    
    if (mkdir(resultPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0) {
        dbgOut(1) << "Results Directory Created" << endl;
    }
    
    std::vector<string> inputFiles = loadStringFile(inputPath);
    std::vector<string> testFiles = loadStringFile(testPath);
    std::vector<float> params = loadParametersFile(parametersFile);

    for (int i = 0 ; i < inputFiles.size() - 1 ; ++i) {
        runExperiments(params, inputFiles[i], testFiles[i], resultPath);
    }
    
    return 0;
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

bool readFile(const std::string &filename, MatMatrix<float> &data, vector<int> &classes, map<int,int> &labels){
    
    if (ArffData::readArffBD(filename, data, labels, classes)) {
        //ArffData::rescaleCols01(data);
        return true;
    }
    
    dbgOut(0) << "Error openning training file" << endl;
    return false;
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

void runExperiments(std::vector<float> params, string filePath, string testPath, string outputPath) {
    
    int numberOfParameters = 7;
    //GRLVQ lvq;
    for (int i=0 ; i < params.size() - 1 ; i += numberOfParameters) { // For each set of parameters
        MatMatrix<float> trainingData;
        vector<int> classes;
        map<int,int> labels;
        readFile(filePath, trainingData, classes, labels);
       
        GRLVQ lvq; 
    
	int nNodes = params[i]; //Number of nodes created in map
	lvq.alpha_tp0 = params[i + 1]; //0.0005; //Learning rate positive
        lvq.alpha_tn0 = params[i + 2]; //0.0005; //Learning rate negative
	lvq.alpha_w0 = params[i + 3]; //0.0001; //Learning rate relevances
	lvq.tau = params[i + 4];
	int epochs = params[i + 5];
        unsigned int seed = params[i + 6];
        
        lvq.min_change = -1;
                             
        //Train LVQ
	trainLVQ(lvq, nNodes, epochs, seed, trainingData, classes, labels);

        string index = std::to_string((i/numberOfParameters));
        classifyTestData(testPath, outputPath + getFileName(testPath) + "_" + index + ".resultsacc", lvq); 
    }
}

void classifyTestData (string testPath, string outputPath, GRLVQ &lvq) {
    MatMatrix<float> testData;
    vector<int> testClasses;
    map<int,int> testLabels;
    readFile(testPath, testData, testClasses, testLabels);
    
    std::ofstream file;
    file.open(outputPath.c_str());
    
    if (!file.is_open()) {
        dbgOut(0) << "Error openning output file" << endl;
        return;
    }

    for (int i = 0 ; i < testData.rows() ; i++) {
        MatVector<float> sample;
        testData.getRow(i, sample);

        std::vector<int> winners;
        winners.push_back(lvq.getWinnerClass(sample));

        for (int j = 0; j < winners.size(); j++) {
            file << i << "\t";
            file << winners[j];
            file << endl;
        }
    }

    file.close();
}

bool trainLVQ(GRLVQ &lvq, int nNodes, int epochs, unsigned int seed, 
        MatMatrix<float> &trainingData, vector<int> &classes, map<int,int> &labels) {
    
    lvq.data = trainingData;
    lvq.vcls = classes;
    
    //Initialize LVQ
    srand(seed);
    lvq.initialize(nNodes, labels.size(), trainingData.cols());
    
    //Print parameters
    dbgOut(2) << std::fixed << std::setprecision(15);
    dbgOut(2) << "Numero de Nodos: " << nNodes << endl;
    dbgOut(2) << "T.A Positiva: " << lvq.alpha_tp0 << endl;
    dbgOut(2) << "T.A Negativa: " << lvq.alpha_tn0 << endl;
    dbgOut(2) << "T.A Peso: " << lvq.alpha_w0 << endl;
    dbgOut(2) << "T. Decaimento: " << lvq.tau << endl;
    dbgOut(2) << "Numero de Epocas: " << epochs << endl;
    dbgOut(2) << "Semente: " << seed << endl;
    
    //Train LVQ
    lvq.tmax = epochs;
    lvq.trainning(lvq.tmax);
    
    return true;
}

