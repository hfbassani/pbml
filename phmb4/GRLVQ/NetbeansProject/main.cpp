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

#define RG_MIN 0 
#define RG_MAX 1 

#define NNODES 0
#define ATP 1
#define ATN 2
#define AW 3
#define TAU 4
#define EPOCHS 5
#define SEED 6

using namespace std;
std::vector<float> loadParametersFile(string path);
std::vector<string> loadStringFile(string path);
bool readFile(const std::string &filename, MatMatrix<float> &data, vector<int> &classes, map<int,int> &labels);
void runLHS();
void runExperiments(std::vector<float> params, string filePath, string outputPath);
void runTrainTestExperiments(std::vector<float> params, string filePath, string testPath, string outputPath);
bool trainLVQ(GRLVQ &lvq, int nNodes, int epochs, unsigned int seed, MatMatrix<float> &trainingData, vector<int> &classes, map<int,int> &labels);

int classIndex = -1; // Class index: 0 firts column; -1 last column
int nCls = 2; //Number of classes in data
int top = 5; //Numero das interacoes de SNPs avaliados atraves da acuracia

string filein;
//string filein = "../Data/Moore/threeway/20SNPs/800/best1.txt";

string fileout;
//string fileout = "output/Velez/1000SNPs/relevances-1000-65.800.0-";

void printContigencyTable(MatVector<int> &colIndexes, MatVector<float> &values, MatMatrix<float> &data, vector<int> &classes){
    MatMatrix<int> result(pow(values.size(), colIndexes.size()), colIndexes.size() + 2);
    result.fill(0);
    
    //Fill first cols
    int alter=1;
    for (int j=0; j < result.cols()-2; j++) {
        int v=0;
        for (int i=0; i < result.rows(); i++) {
            result[i][j] = values[v];
            if ((i+1)%alter==0) v++;
            if (v>values.size()-1) v = 0;
        }
        alter = alter*values.size();
    }
    
    //Fill case/ctrl cols from data
    for (int r=0; r<data.rows(); r++) {
        for (int i=0; i < result.rows(); i++) {
            bool match=true;
            for (int j=0; j < result.cols()-2; j++) {
                if (data[r][colIndexes[j]]!=result[i][j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
               if (classes[r]==0) {
                   result[i][result.cols()-2]++;
               }
               else {
                   result[i][result.cols()-1]++;
               }
               break;
            }
        }
    }
    
    //calculate Entropy
    double entropy = 0; double pmf = 0;
    
    for (int i=0; i < result.rows(); i++) {
        for (int j=1; j < result.cols(); j++)  {
            
            if (result[i][j] != 0){
                pmf = (double)result[i][j]/data.rows();
                entropy = - entropy + ((pmf) * log(pmf));
                
            }
        }
        dbgOut(2) << endl;
    }
    dbgOut(2) << endl;
    dbgOut(2) << entropy << endl;
    
    
    //calculate confusion matrix
    MatMatrix<float> m_confusion(2,2);
    m_confusion.fill(0);
    int t_case = 0;
    int t_control = 0;
    
    for (int i=0; i < result.rows(); i++) {
        t_case = result[i][result.cols()-1];
        t_control = result[i][result.cols() - 2];
                
        if (t_case > t_control) {
            m_confusion[1][1] += t_case;
            m_confusion[1][0] += t_control;
        }
        if (t_control > t_case) {
            m_confusion[0][0] += t_control;
            m_confusion[0][1] += t_case;
        }
    }
    
    //calculate metrics
    float tp, tn, fp, fn = 0;
    tp = m_confusion[1][1];
    tn = m_confusion[0][0];
    fn = m_confusion[1][0];
    fp = m_confusion[0][1];
    
    //Accuracy
    float acc = 0;
    acc = (tn + tp)/(tn + tp + fn + fp);
    
    //Precision
    float precision = 0;
    precision = (tp)/(tp+fp);
    
    //Recall
    float recall = 0;
    recall = tp/(tp+fn);
    
    //F1
    float f1 = 0;
    f1 = (2*tp)/((2*tp) + fp + fn);
    
    //Fmeasure
    float odd = 0;
    odd = (tp/fp)/(fn/tn);
    
    float oddsq = 0;
    oddsq = sqrt((1/tp)+(1/fn)+(1/fp)+(1/tn));
    
    //
    
    //print confusion matrix
    for (int i=0; i < m_confusion.rows(); i++) {
        for (int j=0; j < m_confusion.cols(); j++)  {
            dbgOut(0) << m_confusion[i][j] << "\t";
        }
        dbgOut(0) << endl;
    }
    dbgOut(0) << acc << "\t" << precision << "\t" << recall << "\t" << f1 <<  "\t";
    dbgOut(0) << tp << "\t" << tn << "\t" << fp << "\t" << fn;
    dbgOut(0) << endl;

}

void printInteractionsAccuracy(int top, MatMatrix<float> &trainingData, GRLVQ &lvq){
    MatVector<int> colIndexes;
    
    for (int i=0; i<top; i++){
        //colIndexes.append(i);
        dbgOut(0) << "[" << i << "]: " << lvq.weight[i] << endl;
        //printContigencyTable(colIndexes, values, trainingDataOrig, classes);
        
    }
    
}

bool printNMax(GRLVQ &lvq, int nNodes, int i, int epochs, unsigned int seed) {
    int n = lvq.weight.size();
    MatVector<float> relevances = lvq.weight;
    
    std::stringstream filenameout;
    filenameout << fileout << i << ".txt";
    
    ofstream file(filenameout.str().c_str());
    if(!file.is_open()){
        dbgOut(0) << "Error openning training file" << endl;
        return false;
    }
    dbgOut(2) << "Relevancias em: " << filenameout.str().c_str() << endl;
    
    //Print parameters in file
    file << std::fixed << std::setprecision(7);
    file << "Numero de Nodos:\t" << lvq.meshNodeSet.size() << endl;
    file << "T.A Positiva:\t" << lvq.alpha_tp0 << endl;
    file << "T.A Negativa:\t" << lvq.alpha_tn0 << endl;
    file << "T.A Peso:\t" << lvq.alpha_w0 << endl;
    file << "T. Decaimento:\t" << lvq.tau << endl;
    file << "Numero de Epocas:\t" << epochs  << endl;
    file << "Semente:\t" << seed << endl << endl;
    
    //Print n max relevances
    float max = -1;
    int index = 0;
    for (int i=0; i<n; i++) {
        for (int c=0; c<relevances.size(); c++) {
            if (relevances[c]>max) {
                max = relevances[c];
                index = c;
            }
        }
        
        dbgOut(2) << index << "\t" << max << "\n";
        file << index << "\t" << max << endl;
        relevances[index] = -1;
        max = -1;
    }
    dbgOut(2) << endl;
    
    file.close();
    
    return true;
}

void printRank(int n, MatVector<float> rel){
    
    int count;
    int countSoma;
    
    countSoma = 0;
    for(int i=0; i<n; i++){
        count = 0;
        for(int j=0; j<rel.size(); j++){
            if(rel[i] < rel[j]){
                count++;
            }
        }
        dbgOut(0) << "[" << i << "]: " << rel[i] << " " << "[" << count << "]" << endl;
        countSoma = countSoma + count;
    }
    dbgOut(2) << countSoma << endl;
    dbgOut(0) << endl;
}

double round(double n, unsigned d){
    return floor(n * pow(10., d) + .5) / pow(10., d);
} 

int main(int argc, char** argv) {

    dbgThreshold(1);

    dbgOut(1) << "Running GRLVQ" << endl;

    string inputPath = "";
    string testPath = "";
    string resultPath = "";
    string parametersFile = "";
    bool runTrainTest = false;
        
    int c;
    while ((c = getopt(argc, argv, "i:t:r:p:c")) != -1) {

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
            case 'c':
                runTrainTest = true;
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
        if(!runTrainTest) {
//            runExperiments(params, inputFiles[i], resultPath, supervisionRate, isSubspaceClustering, isFilterNoise, isSorted);
        } else {
//            runTestTrainExperiments(params, inputFiles[i], testFiles[i], resultPath, supervisionRate, isSubspaceClustering, isFilterNoise, isSorted);
        }
    }
    
    filein = argv[1];
    fileout = argv[2];
    
    srand(0);
    runLHS();
    
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
    
    if (ArffData::readArff(filename, data, labels, classes)) {
        ArffData::rescaleCols01(data);
        return true;
    }
    
    dbgOut(0) << "Error openning training file" << endl;
    return false;
}

void runLHS(){
        MatMatrix<float> trainingData;
        vector<int> classes;
        map<int,int> labels;
        std::string fileName = " ";
	
        MatMatrix<double> ranges(7, 2);
        ranges[NNODES][RG_MIN] = 10; ranges[NNODES][RG_MAX] = 30; //2 a 10
	ranges[ATP][RG_MIN] = 0.4; ranges[ATP][RG_MAX] = 0.5;      
	ranges[ATN][RG_MIN] = 0.01; ranges[ATN][RG_MAX] = 0.05; //Valores Percentuais de ATP   
	ranges[AW][RG_MIN] = 0.15; ranges[AW][RG_MAX] = 0.2; 
	ranges[TAU][RG_MIN] = 0.000001; ranges[TAU][RG_MAX] = 0.00002; //ideal 000001 to 0.000005
	ranges[EPOCHS][RG_MIN] = 5000; ranges[EPOCHS][RG_MAX] = 10000;
        ranges[SEED][RG_MIN] = 0; ranges[SEED][RG_MAX] = 10000;

	int L = 100; //Number of parameters sorted
	MatMatrix<double> lhs;
	LHS::getLHS(ranges, lhs, L); //Create matrix lhs with parameters
        
        //Open File
        readFile(fileName, trainingData, classes, labels);
        
        
	//GRLVQ lvq;
	for (int i=0;i<L;i++) { // For each set of parameters
                GRLVQ lvq;
		int nNodes = round(lhs[i][NNODES], 7); //Number of nodes created in map
		lvq.alpha_tp0 = round(lhs[i][ATP], 7); //0.0005; //Learning rate positive
                lvq.alpha_tn0 = round(lhs[i][ATN], 7); //0.0005; //Learning rate negative
		lvq.alpha_w0 = round(lhs[i][AW], 7); //0.0001; //Learning rate relevances
		lvq.tau = round(lhs[i][TAU], 7);
                lvq.min_change = 0.01;
		int epochs = lhs[i][EPOCHS];
                unsigned int seed = (unsigned int) lhs[i][SEED];
                               
                //Train LVQ
		trainLVQ(lvq, nNodes, epochs, seed, trainingData, classes, labels);

                //Print Results
                printNMax(lvq, nNodes, i, epochs, seed);
                            
                //Print Positions of Revelant SNPs
                printRank(5, lvq.weight);
                
	}
}

void runTrainTestExperiments(std::vector<float> params, string filePath, string testPath, string outputPath) {
    
}

void runExperiments(std::vector<float> params, string filePath, string outputPath) {
    MatMatrix<float> trainingData;
    vector<int> classes;
    map<int,int> labels;
	
    readFile(filePath, trainingData, classes, labels);
       
    GRLVQ lvq; 
    
    int numberOfParameters = 9;
    //GRLVQ lvq;
    for (int i=0 ; i < params.size() - 1 ; i++) { // For each set of parameters
	int nNodes = params[i]; //Number of nodes created in map
	lvq.alpha_tp0 = params[i + 2]; //0.0005; //Learning rate positive
        lvq.alpha_tn0 = params[i + 3]; //0.0005; //Learning rate negative
	lvq.alpha_w0 = params[i + 4]; //0.0001; //Learning rate relevances
	lvq.tau = params[i + 5];
        lvq.min_change = params[i + 6];
	int epochs = params[i + 7];
        unsigned int seed = params[i + 8];
                               
        //Train LVQ
	trainLVQ(lvq, nNodes, epochs, seed, trainingData, classes, labels);

        //Print Results
        printNMax(lvq, nNodes, i, epochs, seed);
                            
        //Print Positions of Revelant SNPs
        printRank(5, lvq.weight);
                
    }
}

bool trainLVQ(GRLVQ &lvq, int nNodes, int epochs, unsigned int seed, MatMatrix<float> &trainingData, vector<int> &classes, map<int,int> &labels) {
    
    lvq.data = trainingData;
    lvq.vcls = classes;
    
    //Initialize LVQ
    srand(seed);
    lvq.initialize(nNodes, nCls, trainingData.cols());
    
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

