/*
 * File:   main.cpp
 * Author: hans
 *
 * Created on 11 de Outubro de 2010, 07:25
 */
#define NO_INVALID_DIMENSION_SIZE
//#define PRINT_CLUSTER

#include <stdlib.h>
#include <fstream>
#include <algorithm>
#include <time.h>
#include <limits>
#include "MatMatrix.h"
#include "MatVector.h"
#include "DebugOut.h"
#include "ClusteringMetrics.h"
#include "ArffData.h"
#include "ClusteringSOM.h"
#include "VILMAP.h"
#include "unistd.h"
#include "TextToPhoneme.h"
#include "MyParameters/MyParameters.h"
#include "OutputMetrics/OutputMetrics.h"
#include <string>
#include <sys/stat.h>

using namespace std;

void createTrainingTestFiles(int d_min, int d_max, std::ifstream &file, std::string &dictionary, std::string &featuresDict, const std::string &filename);
void createInputData(std::vector<FeaturesVector> &data, int inputCount, MatMatrix<float> &mdata, std::map<int, int> &groupLabels, std::vector<int> &groups);
void createPhonemaData(std::string &featuresDict, MatMatrix<float> &data);

int loadTrueFeatureMatrix(std::ifstream &file, std::string &dictionary, std::string &featuresDict, std::vector<FeaturesVector> &data, const std::string &filename);
void loadFalseFeatureMatrix(std::string &featuresDict, std::vector<FeaturesVector> &data, const std::string &filename, int phonemesNum);

MatMatrix<float> loadFalseData(int tam, int fileNumber);
MatMatrix<float> loadTrueData(int tam, int fileNumber);
MatMatrix<float> loadTestData(int tam);
MatMatrix<float> loadDataFromPath(string path);

void runCompleteTest(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, int paramsNumber, std::string &featuresDict, OutputMetrics outputM);
void runTestAfterTraining(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, int paramsNumber, std::string &featuresDict, OutputMetrics outputM);
void runStudyOfCase(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, int paramsNumber, std::string &featuresDict, OutputMetrics outputM);
void runStudyOfCaseAfterTraining(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, std::string &featuresDict, OutputMetrics outputM);
void runStudyOfCaseAfterTrainingBrentDataBase(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, std::string &featuresDict, OutputMetrics outputM, int exp);
void learningTest(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, std::string &featuresDict, OutputMetrics outputM);

void createParametersFile(MyParameters *params);
void createParametersFiles(MyParameters *params);

std::vector<float> loadParametersFile(int number);
std::vector<float> loadParametersFile();

void loadTrueDataInMemory();
void loadFalseDataInMemory();

void runTimeSeriesMotifDiscovery(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, int paramsNumber, std::string &featuresDict, OutputMetrics outputM);
void runTestAfterTrainingTimeSeries(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, int paramsNumber, std::string &featuresDict, OutputMetrics outputM);

vector<string> splitString(string str, char delimiter);
void createInputDataFromTimeSeries(int dimension, string path);
MatMatrix<float> loadInputDataFromTimeSeries(int dimension, string path);
MatMatrix<float> createOneInputDataFromTimeSeries(int dimension, string path);

std::vector<MatMatrix<float> > v_false[100]; //Vector of true data
std::vector<MatMatrix<float> > v_true[100]; //Vector of false data

int main(int argc, char** argv) {

    MyParameters params; //Teste de Parâmetros
    OutputMetrics outputM; //Classe que escreve a saída das métricas
    int seed = 0;
    dbgThreshold(1);
    VILMAP som(1);
    SOM<DSNode> *dssom = (SOM<DSNode>*) & som;
    ClusteringMeshSOM clusteringSOM(dssom);

    string dictionary = "dictionary/c1-dict.csv";
    string featuresDict = "dictionary/PhonemeFeatures.csv";
    string filename = "";

    //Inicializar parametros
    som.maxNodeNumber = 20;
    som.e_b = 0.0005;
    som.e_n = 0.000001;
    som.dsbeta = 0.001;
    som.epsilon_ds = 0.0;
    som.minwd = 0.5;
    som.age_wins = 10;
    som.lp = 0.061; //0.175;
    som.a_t = 0.95; //Limiar de atividade .95
    int epocs = 10000;
    clusteringSOM.setFilterNoise(false);
    clusteringSOM.setIsSubspaceClustering(false);
    som.d_max = 6; //Tamanho máximo de entrada
    som.d_min = 2; //tamanho mínimo de entrada
    int c;
    while ((c = getopt(argc, argv, "f:n:t:e:g:m:s:p:w:i:l:v:D:d:r:Po")) != -1)
        switch (c) {
            case 'f':
                filename.assign(optarg);
                break;
            case 'n':
            case 't':
                epocs = atoi(optarg);
                break;
            case 'e':
                som.e_b = atof(optarg);
                break;
            case 'g':
                som.e_n = atof(optarg);
                break;
            case 'm':
                som.maxNodeNumber = atoi(optarg);
                break;
            case 's':
                //DS
                som.dsbeta = atof(optarg);
                break;
            case 'p':
                som.epsilon_ds = atof(optarg);
                break;
            case 'w':
                som.minwd = atof(optarg);
                break;
            case 'i':
                som.age_wins = atof(optarg);
                break;
            case 'l':
                som.lp = atof(optarg);
                break;
            case 'v':
                som.a_t = atof(optarg);
                break;
            case 'D':
                som.d_max = atoi(optarg);
                break;
            case 'd':
                som.d_min = atoi(optarg);
                break;
            case 'P':
                clusteringSOM.setIsSubspaceClustering(false);
                break;
            case 'r':
                seed = atoi(optarg);
                break;
            case 'o':
                clusteringSOM.setFilterNoise(false);
                break;
            case '?':
                if (optopt == 'f')
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint(optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr,
                        "Unknown option character `\\x%x'.\n",
                        optopt);
                return 1;
        }


    filename = "input/c1_cat.txt";

    if (filename == "") {
        dbgOut(0) << "option -f [filename] is required" << endl;
        return -1;
    }

    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        dbgOut(0) << "Error openning file: " << filename << endl;
        return false;
    }

    dbgOut(1) << "Running VILMAP for file: " << filename << endl;
    mkdir("output/", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    srand(time(0));
    srand(seed);
    //dbgOut(1) << "Loading False data in Memory..." << endl;
    //loadFalseDataInMemory(); //[i][j] i - fileNumber // j - InputSize
    //dbgOut(1) << "Loading True data in Memory..." << endl;
    //loadTrueDataInMemory();
    dbgOut(1) << "Running test" << endl;



    //runTimeSeriesMotifDiscovery(&som, clusteringSOM, dssom, epocs, featuresDict, outputM);

    //runTestAfterTrainingTimeSeries(&som, clusteringSOM, dssom, epocs, featuresDict, outputM);


    //learningTest(&som, clusteringSOM, dssom, featuresDict, outputM);

    /*//////Gerando os arquivos na mão

    int tam = 6;
    std::vector<FeaturesVector> phonemesData;
    //std::vector<FeaturesVector> phonemesFalseData;
    int phonemesNum = loadTrueFeatureMatrix(file, dictionary, featuresDict, phonemesData, filename + ".new");
    //loadFalseFeatureMatrix(featuresDict, phonemesFalseData, "input/phonemes_falseFile", phonemesNum);
    dbgOut(1) << "init true" << endl;
    MatMatrix<float> data;
    //std::vector<int> groups;
    //std::map<int, int> groupLabels;
    //createInputData(phonemesData, tam, data, groupLabels, groups);
    string name_true = "trueData_" + std::to_string(tam) + "_arq_c1";
    std::ofstream file_true;
    file_true.open(name_true.c_str());
    for (int ph = 0; ph < phonemesData.size(); ph++) {
        if (phonemesData[ph].cols() <= 6) {
            for (int j = 0; j < phonemesData[ph].cols(); j++) {
                for (int i = 0; i < phonemesData[ph].rows(); i++) {
                    file_true << phonemesData[ph][i][j] << "\t";
                }
                
            }
            file_true << "\n";
        }
    }
    //dbgOut(1) << "init false" << endl;
    //string name_false = "falseData_" + std::to_string(tam) + "_arq_c1";
    //std::ofstream file_false;
    //file_false.open(name_false.c_str());

    //False file
    //MatMatrix<float> dataFalse;
    //createInputData(phonemesFalseData, tam, dataFalse, groupLabels, groups);
    //for (int i = 0; i < dataFalse.rows(); i++) {
    //    for (int j = 0; j < dataFalse.cols(); j++) {
    //        file_false << dataFalse[i][j] << "\t";
    //    }
    //    file_false << "\n";
    //}

    /*/////////

    //createParametersFile(&params); //createTrainingTestFiles(som.d_min, som.d_max, file, dictionary, featuresDict, "phonemes_" + filename);
    runCompleteTest(&som, clusteringSOM, dssom, epocs, featuresDict, outputM);
    //runStudyOfCase(&som, clusteringSOM, dssom, epocs, featuresDict, outputM);

    //runTestAfterTraining(&som, clusteringSOM, dssom, epocs, featuresDict, outputM);

    //runStudyOfCaseAfterTraining(&som, clusteringSOM, dssom, featuresDict, outputM);

    //runStudyOfCaseAfterTrainingBrentDataBase(&som, clusteringSOM, dssom, featuresDict, outputM, epocs);

    dbgOut(1) << "Done." << endl;
}

void runCompleteTest(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, int paramsNumber, std::string &featuresDict, OutputMetrics outputM) {
    int experiment = 1;
    int fileNumber = 0;
    string filename = "";
    std::vector<float> params = loadParametersFile();
    //for (fileNumber = paramsNumber - 1; fileNumber < paramsNumber; fileNumber++) {
    //    filename = "sentences_" + std::to_string(fileNumber) + ".txt";
    //    dbgOut(1) << "FileNumber = " << std::to_string(fileNumber) << endl;

    for (int i = 0; i < params.size(); i += 10) {
        som->reset();
        som->a_t = params[i];
        som->lp = params[i + 1];
        som->dsbeta = params[i + 2];
        som->e_b = params[i + 4];
        som->e_n = params[i + 5];
        som->epsilon_ds = params[i + 6];
        som->minwd = params[i + 7];
        //experiment = (i / 7) + ((paramsNumber - 1) * 17);
        experiment = (i / 10);
        MatMatrix<int> taxaTrue, taxaFalse; // 0 - Ativaçoes totais // 1 - Ativações reconhecidas // 2 - Ativações Não reconhecidas
        dbgOut(1) << "f-" << fileNumber << " e-" << experiment;
        if(experiment < 95){
            continue;
        }
        //Faz o treinamento e teste para a quantidade de dimensões solicitadas com taxas
        som->d_max = 6;
        som->d_min = 2;
        for (int i = som->d_min; i <= som->d_max; i++) {
            //Taxa de true positive
            MatMatrix<float> data = loadTrueData(i, fileNumber);
            clusteringSOM.setData(data);
            som->resetSize(clusteringSOM.getInputSize());
            clusteringSOM.trainSOM(1); // 1 - Epocs
            //taxaTrue.concatRows(clusteringSOM.writeClusterResultsReadable("output/result_" + std::to_string(i) + "_" + filename, data, featuresDict, dssom, som->a_t));

            //Taxa de false negative
            //MatMatrix<float> dataFalse = loadFalseData(i, fileNumber);
            //clusteringSOM.setData(dataFalse);
            //taxaFalse.concatRows(clusteringSOM.writeClusterResultsReadable("output/false_" + std::to_string(i) + "_" + filename, dataFalse, featuresDict, dssom, som->a_t));
            if (i == som->d_max) {
                som->saveSOM("networks1/som_arq_" + std::to_string(fileNumber) + "_exp_" + std::to_string(experiment) + "_TE_" + std::to_string(i));
            }

        }
        //outputM.PATH = "output/metrics" + std::to_string(paramsNumber) + "/";
        //outputM.outputWithParamsFiles(som, experiment, taxaTrue, taxaFalse, fileNumber);
        dbgOut(1) << std::to_string(experiment + 1) << "% do arquivo " << std::to_string(fileNumber) << endl;
        som->reset();
    }
    //}
}

void runTestAfterTraining(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, int paramsNumber, std::string &featuresDict, OutputMetrics outputM) {
    string testname = "sentences.txt";
    som->d_max = 6;
    som->d_min = 2;
    int fileNumber = 0;
    int experiment;
    for (fileNumber = paramsNumber - 1; fileNumber < paramsNumber; fileNumber++) { // For para arquivos
        for (experiment = 0; experiment <= 99; experiment++) { // For para experimentos
            MatMatrix<int> taxaTrue, taxaFalse; // 0 - Ativaçoes totais // 1 - Ativações reconhecidas // 2 - Ativações Não reconhecidas
            dbgOut(1) << "f-" << fileNumber << " e-" << experiment;
            //Testa com todos os arquivos de entrada depois que arede já foi treinada
            som->readSOM("networks1/som_arq_" + std::to_string(fileNumber) + "_exp_" + std::to_string(experiment) + "_TE_" + std::to_string(6));


            for (int i = som->d_min; i <= som->d_max; i++) { // For para tamanhos de entrada

                //Taxa de true positive
                MatMatrix<float> data = loadTrueData(i, fileNumber);
                clusteringSOM.setData(data);
                som->resetSize(clusteringSOM.getInputSize());
                taxaTrue.concatRows(clusteringSOM.writeClusterResultsReadable("output/result_" + std::to_string(i) + "_" + testname, data, featuresDict, dssom, som->a_t));

                //Taxa de false negative
                MatMatrix<float> dataFalse = loadFalseData(i, fileNumber);
                clusteringSOM.setData(dataFalse);
                som->resetSize(clusteringSOM.getInputSize());
                taxaFalse.concatRows(clusteringSOM.writeClusterResultsReadable("output/false_" + std::to_string(i) + "_" + testname, dataFalse, featuresDict, dssom, som->a_t));

            }

            outputM.PATH = "output/metrics" + std::to_string(paramsNumber + 10) + "/";
            outputM.outputWithParamsFiles(som, experiment, taxaTrue, taxaFalse, fileNumber);
            dbgOut(1) << std::to_string(experiment) << "% Concluido do arquivo " << fileNumber << endl;


        }

    }
}

void runStudyOfCase(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, int paramsNumber, std::string &featuresDict, OutputMetrics outputM) {
    int experiment = 47; // Número do experimento
    int fileNumber = 0; // Número do arquivo de entrada
    string filename = "c1_cat.txt";
    std::vector<float> params = loadParametersFile();
    int i = experiment * 10;
    //filename = "sentences_" + std::to_string(fileNumber) + ".txt";

    som->a_t = params[i];
    som->lp = params[i + 1];
    som->dsbeta = params[i + 2];
    som->e_b = params[i + 4];
    som->e_n = params[i + 5];
    som->epsilon_ds = params[i + 6];
    som->minwd = params[i + 7];

    MatMatrix<int> taxaTrue, taxaFalse; // 0 - Ativaçoes totais // 1 - Ativações reconhecidas // 2 - Ativações Não reconhecidas
    dbgOut(1) << "f-" << fileNumber << " e-" << experiment << endl;
    //Faz o treinamento e teste para a quantidade de dimensões solicitadas com taxas
    som->d_max = 6;
    for (int i = som->d_min; i <= som->d_max; i++) {
        //Taxa de true positive
        //MatMatrix<float> data = loadTrueData(i, fileNumber);
        dbgOut(1) << "init Dimension = " << i << endl;
        MatMatrix<float> data = loadTrueData(i, fileNumber);
        clusteringSOM.setData(data);
        som->resetSize(clusteringSOM.getInputSize());
        clusteringSOM.trainSOM(1); // 1 - Epocs
        taxaTrue.concatRows(clusteringSOM.writeClusterResultsReadable("output/result_" + std::to_string(i) + "_" + filename, data, featuresDict, dssom, som->a_t));

        //Taxa de false negative
        MatMatrix<float> dataFalse = loadFalseData(i, fileNumber);
        clusteringSOM.setData(dataFalse);
        taxaFalse.concatRows(clusteringSOM.writeClusterResultsReadable("output/false_" + std::to_string(i) + "_" + filename, dataFalse, featuresDict, dssom, som->a_t));
        //if (i == som->d_max) {
        som->saveSOM("networks/som_arq_" + std::to_string(fileNumber) + "_exp_" + std::to_string(experiment) + "_TE_" + std::to_string(i));
        //}

        dbgOut(1) << "end Dimension = " << i << endl;
    }

    outputM.PATH = "output/";
    outputM.outputWithParamsFiles(som, experiment, taxaTrue, taxaFalse, fileNumber);
    dbgOut(1) << std::to_string(experiment) << "% do arquivo " << std::to_string(fileNumber) << endl;

}

void runStudyOfCaseAfterTraining(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, std::string &featuresDict, OutputMetrics outputM) {
    int experiment = 33;
    int fileNumber = 4;
    string filename = "sentences_" + std::to_string(fileNumber) + ".txt";

    dbgOut(1) << "f-" << fileNumber << " e-" << experiment;
    MatMatrix<int> taxaTrue, taxaFalse; // 0 - Ativaçoes totais // 1 - Ativações reconhecidas // 2 - Ativações Não reconhecidas
    //Testa com todos os arquivos de entrada depois que arede já foi treinada
    som->readSOM("networks1/som_arq_" + std::to_string(fileNumber) + "_exp_" + std::to_string(experiment) + "_TE_" + std::to_string(6));

    for (int i = som->d_min; i <= som->d_max; i++) { // For para tamanhos de entrada

        //Taxa de true positive
        MatMatrix<float> data = loadTrueData(i, fileNumber);
        clusteringSOM.setData(data);
        som->resetSize(clusteringSOM.getInputSize());
        taxaTrue.concatRows(clusteringSOM.writeClusterResultsReadable("output/result_" + std::to_string(i) + "_" + filename, data, featuresDict, dssom, som->a_t));

        //Taxa de false negative
        MatMatrix<float> dataFalse = loadFalseData(i, fileNumber);
        clusteringSOM.setData(dataFalse);
        som->resetSize(clusteringSOM.getInputSize());
        taxaFalse.concatRows(clusteringSOM.writeClusterResultsReadable("output/false_" + std::to_string(i) + "_" + filename, dataFalse, featuresDict, dssom, som->a_t));

    }

    outputM.PATH = "output/";
    outputM.outputWithParamsFiles(som, experiment, taxaTrue, taxaFalse, fileNumber);
    dbgOut(1) << std::to_string(experiment) << "% Concluido do arquivo " << fileNumber << endl;
}

void runStudyOfCaseAfterTrainingBrentDataBase(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, std::string &featuresDict, OutputMetrics outputM, int exp) {
    //    Funcao criada para comparar e gerar resultados do artigo
    int experiment = 14;
    int fileNumber = 0;
    MatMatrix<float> data = loadTrueData(6, fileNumber);
    MatMatrix<float> dataFalse = loadFalseData(6, fileNumber);
    string filename = "sentences_" + std::to_string(fileNumber) + ".txt";
    for (experiment = exp; experiment <= exp + 15; experiment++) {
        dbgOut(1) << "f-" << fileNumber << " e-" << experiment;
        MatMatrix<int> taxaTrue, taxaFalse; // 0 - Ativaçoes totais // 1 - Ativações reconhecidas // 2 - Ativações Não reconhecidas
        //Testa com todos os arquivos de entrada depois que arede já foi treinada
        som->readSOM("networks1/som_arq_" + std::to_string(fileNumber) + "_exp_" + std::to_string(experiment) + "_TE_" + std::to_string(6));

        for (int i = 6; i <= 6; i++) { // For para tamanhos de entrada

            //Taxa de true positive

            clusteringSOM.setData(data);
            som->resetSize(clusteringSOM.getInputSize());
            float at_min = 0.60;
            for (int qtd = 0; qtd < 50; qtd++) {
                dbgOut(1) << "|";
                taxaTrue.concatRows(clusteringSOM.writeClusterResultsArticle("output/result_" + std::to_string(i) + "_" + filename, data, featuresDict, dssom, at_min));
                at_min += 0.01;
            }
            std::ofstream file1;
            std::string name = "article/metrics_articleTRUE_" + std::to_string(experiment) + ".results";
            file1.open(name.c_str(), std::ios_base::app);
            for (int row = 0; row < taxaTrue.rows(); row++) {
                int tp = taxaTrue[row][3];
                int fp = taxaTrue[row][4];
                int tn = taxaTrue[row][5];
                int fn = taxaTrue[row][6];
                float precision = tp / (tp + fp + 0.00000000001);
                float recall = tp / (tp + fn + 0.00000000001);
                file1 << "------------------------------------------------------------" << endl;
                file1 << "tp = " << taxaTrue[row][3] << " | fp = " << taxaTrue[row][4] << " | tn = " << taxaTrue[row][5] << " | fn = " << taxaTrue[row][6] << std::endl;
                file1 << "precision = " << precision << " | recall = " << recall << " | f-measure = " << (2 * precision * recall) / (precision + recall) << std::endl;
                file1 << "at_all = " << taxaTrue[row][0] << " | at_know = " << taxaTrue[row][1] << " | at_Unknown = " << taxaTrue[row][2] << endl;

            }
            //Taxa de false negative

            clusteringSOM.setData(dataFalse);
            som->resetSize(clusteringSOM.getInputSize());
            at_min = 0.60;
            for (int qtd = 0; qtd < 50; qtd++) {
                dbgOut(1) << "|";
                taxaFalse.concatRows(clusteringSOM.writeClusterResultsArticle("output/false_" + std::to_string(i) + "_" + filename, dataFalse, featuresDict, dssom, at_min));
                at_min += 0.01;
            }
            std::ofstream file2;
            std::string name2 = "article/metrics_articleFALSE_" + std::to_string(experiment) + ".results";
            file2.open(name2.c_str(), std::ios_base::app);
            for (int row = 0; row < taxaFalse.rows(); row++) {
                int tp = taxaFalse[row][3];
                int fp = taxaFalse[row][4];
                int tn = taxaFalse[row][5];
                int fn = taxaFalse[row][6];
                float precision = tp / (tp + fp + 0.00000000001);
                float recall = tp / (tp + fn + 0.00000000001);
                file2 << "------------------------------------------------------------" << endl;
                file2 << "tp = " << taxaFalse[row][3] << " | fp = " << taxaFalse[row][4] << " | tn = " << taxaFalse[row][5] << " | fn = " << taxaFalse[row][6] << std::endl;
                file2 << "precision = " << precision << " | recall = " << recall << " | f-measure = " << (2 * precision * recall) / (precision + recall) << std::endl;
                file2 << "at_all = " << taxaFalse[row][0] << " | at_know = " << taxaFalse[row][1] << " | at_Unknown = " << taxaFalse[row][2] << endl;

            }
        }
        if (experiment >= 99) {
            break;
        }
    }
}

void learningTest(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, std::string &featuresDict, OutputMetrics outputM) {
    int experiment = 3;
    int fileNumber = 5;
    string filename = "sentences_" + std::to_string(fileNumber) + ".txt";

    dbgOut(1) << "f-" << fileNumber << " e-" << experiment;
    MatMatrix<int> taxaTrue, taxaFalse; // 0 - Ativaçoes totais // 1 - Ativações reconhecidas // 2 - Ativações Não reconhecidas
    //Testa com todos os arquivos de entrada depois que arede já foi treinada
    som->readSOM("networks1/som_arq_" + std::to_string(fileNumber) + "_exp_" + std::to_string(experiment) + "_TE_" + std::to_string(6) + "_#");

    for (int i = som->d_min; i <= som->d_max; i++) { // For para tamanhos de entrada

        //Taxa de true positive
        MatMatrix<float> data = loadTestData(i);
        clusteringSOM.setData(data);
        som->resetSize(clusteringSOM.getInputSize());
        //taxaTrue.concatRows(clusteringSOM.writeClusterResultsReadable("outputTest/result_" + std::to_string(i) + "_" + filename, data, featuresDict, dssom, som->a_t));
        dbgOut(1) << " Dimension = " << i;

        clusteringSOM.writeClusterResultsHTML("outputTest/result_size-" + std::to_string(i) + ".html", data, featuresDict, dssom, som->a_t, i, experiment, fileNumber);

    }

    dbgOut(1) << std::to_string(experiment) << "% Concluido do arquivo " << fileNumber << endl;
}

void createTrainingTestFiles(int d_min, int d_max, std::ifstream &file, std::string &dictionary, std::string &featuresDict, const std::string &filename) {

    //Gerar arquivos de treinamento e teste
    std::vector<FeaturesVector> phonemesData;
    std::vector<FeaturesVector> phonemesFalseData;
    int phonemesNum = loadTrueFeatureMatrix(file, dictionary, featuresDict, phonemesData, filename);
    loadFalseFeatureMatrix(featuresDict, phonemesFalseData, "input/phonemes_falseFile", phonemesNum);
    std::vector<int> groups;
    std::map<int, int> groupLabels;

    for (int i = d_min; i <= d_max; i++) {
        string name_true = "input/trueData_" + std::to_string(i) + "_arq_";
        std::ofstream file_true;
        file_true.open(name_true.c_str());

        string name_false = "input/falseData_" + std::to_string(i) + "_arq_";
        std::ofstream file_false;
        file_false.open(name_false.c_str());

        //True file
        MatMatrix<float> data;
        createInputData(phonemesData, i, data, groupLabels, groups);
        for (int i = 0; i < data.rows(); i++) {
            for (int j = 0; j < data.cols(); j++) {
                file_true << data[i][j] << "\t";
            }
            file_true << "\n";
        }

        //False file
        MatMatrix<float> dataFalse;
        createInputData(phonemesFalseData, i, dataFalse, groupLabels, groups);
        for (int i = 0; i < dataFalse.rows(); i++) {
            for (int j = 0; j < dataFalse.cols(); j++) {
                file_false << dataFalse[i][j] << "\t";
            }
            file_false << "\n";
        }
    }

}

int loadTrueFeatureMatrix(std::ifstream &file, std::string &dictionary, std::string &featuresDict, std::vector<FeaturesVector> &data, const std::string &filename) {
    TextToPhoneme ttp;
    ttp.loadDictionary(dictionary);
    PhonemesToFeatures pf;
    pf.loadPhonemeFeatures(featuresDict, 12);
    std::string text;
    std::string phonemes;
    FeaturesVector features;
    std::ofstream fileOutput;
    fileOutput.open(filename.c_str());
    MatVector<std::string> output;
    int phonemesNum = 0;
    while (!file.eof()) {
        getline(file, text);
        if (text.length() > 0) {
            if (ttp.translateWords(text, phonemes)) {
                pf.translatePhonemesFeatures(phonemes, features);
                data.push_back(features);
                output.append(text + " -> " + phonemes + " -> " + features.toString());
                phonemesNum += features.cols();
            } else
                dbgOut(0) << "Unknown word: " << text << endl;
        }
    }

    for (int i = 0; i < output.size(); i++) {
        fileOutput << output[i];
        fileOutput << endl;
    }

    fileOutput.close();

    dbgOut(0) << phonemesNum << endl;
    return phonemesNum;
}

void loadFalseFeatureMatrix(std::string &featuresDict, std::vector<FeaturesVector> &data, const std::string &filename, int phonemesNum) {
    PhonemesToFeatures pf;
    pf.loadPhonemeFeatures(featuresDict, 12);
    std::string phoneme;
    std::string phonemes;
    FeaturesVector features;
    std::ofstream fileOutput;
    fileOutput.open(filename.c_str());
    MatVector<std::string> output;
    for (int i = 0; i < phonemesNum; i++) {
        pf.getRandomPhoneme(phoneme);
        phonemes += phoneme + ' ';
    }
    if (!phonemes.empty()) {
        output.append(phonemes);
        pf.translatePhonemesFeatures(phonemes, features);
        data.push_back(features);
    } else {
        dbgOut(0) << "Error in dict: " << phonemes << endl;
    }

    for (int i = 0; i < output.size(); i++) {
        fileOutput << output[i] << endl;
    }

    fileOutput.close();
}

void createInputData(std::vector<FeaturesVector> &data, int inputCount, MatMatrix<float> &mdata, std::map<int, int> &groupLabels, std::vector<int> &groups) {

    int group = 0;

    for (int i = 0; i < data.size(); i++) {
        FeaturesVector fb = data[i];
        MatVector<float> vect(inputCount * fb.rows());

        for (int j = 0; j < inputCount + fb.cols(); j++) {
            int start = max((int) inputCount - j - 1, 0);
            int end = min((int) inputCount - 2 - j + (int) fb.cols(), (int) inputCount - 1);

            for (int k = 0; k < inputCount; k++) {
                for (int r = 0; r < fb.rows(); r++) {
                    int l = k * fb.rows() + r;
                    if (k >= start && k <= end) {
                        int index = j - inputCount + 1 + k;
                        vect[l] = fb[r][index];
                    } else
                        //vect[l] = fb.min() + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(fb.max()-fb.min())));//random no range da série de entrada
                        vect[l] = 0;
                }
            }
            mdata.concatRows(vect);
            groups.push_back(group);
        }
        group++;
    }

    for (int i = 0; i < data.size(); i++)
        groupLabels[i] = i;
}

void createPhonemaData(std::string &featuresDict, MatMatrix<float> &data) {
    PhonemesToFeatures pf;
    std::string phonemes;
    pf.loadPhonemeFeatures(featuresDict, 12);
    Features features;
    features.resize(12);
    MatVector<float> colOfData;
    data.getCol(0, colOfData);

    for (int i = 0; i < colOfData.size(); i++) {
        MatVector<float> rowOfData;
        data.getRow(i, rowOfData);
        for (int begin = 0, end = 11; end < 48; begin += 12, end += 12) {
            features.copy(rowOfData, begin, end);
            pf.translateFeaturesPhoneme(features, phonemes);
            dbgOut(1) << phonemes << " ";
        }
        dbgOut(1) << endl;
    }

}

MatMatrix<float> loadFalseData(int tam, int fileNumber) {
    MatMatrix<float> mat;
    std::ifstream inputFile("input/falseData_"+ std::to_string(tam) +"_arq_c1");
    if (!inputFile.is_open()) {
        dbgOut(0) << "Error openning inputFile. "  << endl;
    }
    std::string text;
    std::string temp = "";
    MatVector<float> output_vect;
    while (!inputFile.eof()) {
        getline(inputFile, text);
        if (text.size() > 0) {
            for (int i = 0; i < text.size(); i++) {
                if (text[i] != '\t') {
                    temp += text[i];
                } else {
                    output_vect.append(std::stof(temp));
                    temp = "";
                }
            }
        }
        if (output_vect.size() > 0) {
            mat.concatRows(output_vect);
            output_vect.clear();
        }
    }
    return mat;
}

MatMatrix<float> loadTrueData(int tam, int fileNumber) {
    MatMatrix<float> mat;
    std::ifstream inputFile("input/trueData_"+ std::to_string(tam) +"_arq_c1");
    if (!inputFile.is_open()) {
        dbgOut(0) << "Error openning inputFile. "  << endl;
    }
    std::string text;
    std::string temp = "";
    MatVector<float> output_vect;
    while (!inputFile.eof()) {
        getline(inputFile, text);
        if (text.size() > 0) {
            for (int i = 0; i < text.size(); i++) {
                if (text[i] != '\t') {
                    temp += text[i];
                } else {
                    output_vect.append(std::stof(temp));
                    temp = "";
                }
            }
        }

        if (output_vect.size() > 0) {
            if (output_vect.size() < 72) {//para não produzir dimensão inválida. Fazer tratamento antes de colocar como estímulo na rede
                for (int i = output_vect.size(); i < 72; i++) {
                    output_vect.append(5);
                }
            }
            mat.concatRows(output_vect);
            output_vect.clear();
        }
    }
    return mat;
}

MatMatrix<float> loadDataFromPath(std::string path) {
    MatMatrix<float> mat;
    std::ifstream inputFile(path);
    std::string text;
    std::string temp = "";
    MatVector<float> output_vect;
    while (!inputFile.eof()) {
        getline(inputFile, text);
        if (text.size() > 0) {
            for (int i = 0; i < text.size(); i++) {
                if (text[i] != '\t') {
                    temp += text[i];
                } else {
                    output_vect.append(std::stof(temp));
                    temp = "";
                }
            }
        }
        if (output_vect.size() > 0) {
            mat.concatRows(output_vect);
            output_vect.clear();
        }
    }
    return mat;
}

MatMatrix<float> loadInputDataFromTimeSeries(int dimension, string path) {
    MatMatrix<float> mat;
    std::ifstream inputFile(path + std::to_string(dimension) + "_arq_0.txt");
    std::string text;
    std::string temp = "";
    MatVector<float> output_vect;
    while (!inputFile.eof()) {
        getline(inputFile, text);
        if (text.size() > 0) {
            for (int i = 0; i < text.size(); i++) {
                if (text[i] != '\t') {
                    temp += text[i];
                } else {
                    output_vect.append(std::stof(temp));
                    temp = "";
                }
            }
        }
        if (output_vect.size() > 0) {
            mat.concatRows(output_vect);
            output_vect.clear();
        }
    }
    return mat;
}

MatMatrix<float> loadTestData(int tam) {
    MatMatrix<float> mat;
    std::ifstream inputFile("TestData/trueData_" + std::to_string(tam) + "_arq_#");
    std::string text;
    std::string temp = "";
    MatVector<float> output_vect;
    while (!inputFile.eof()) {
        getline(inputFile, text);
        if (text.size() > 0) {
            for (int i = 0; i < text.size(); i++) {
                if (text[i] != '\t') {
                    temp += text[i];
                } else {
                    output_vect.append(std::stof(temp));
                    temp = "";
                }
            }
        }
        if (output_vect.size() > 0) {
            mat.concatRows(output_vect);
            output_vect.clear();
        }
    }
    return mat;
}

std::vector<float> loadParametersFile(int number) {
    string name1 = "input/MyParameters" + std::to_string(number);
    std::ifstream file(name1.c_str());
    std::string text;
    std::vector<float> params;
    while (!file.eof()) {
        getline(file, text);
        params.push_back(std::atof(text.c_str()));
    }
    return params;
}

std::vector<float> loadParametersFile() {
    string name1 = "input/ParamsRapha_0";
    std::ifstream file(name1.c_str());
    std::string text;
    std::vector<float> params;
    while (!file.eof()) {
        getline(file, text);
        params.push_back(std::atof(text.c_str()));
    }
    return params;
}

void createParametersFiles(MyParameters * params) {
    string name1 = "input/MyParameters1";
    std::ofstream file1;
    file1.open(name1.c_str());

    string name2 = "input/MyParameters2";
    std::ofstream file2;
    file2.open(name2.c_str());

    string name3 = "input/MyParameters3";
    std::ofstream file3;
    file3.open(name3.c_str());

    string name4 = "input/MyParameters4";
    std::ofstream file4;
    file4.open(name4.c_str());

    string name5 = "input/MyParameters5";
    std::ofstream file5;
    file5.open(name5.c_str());

    string name6 = "input/MyParameters6";
    std::ofstream file6;
    file6.open(name6.c_str());
    cout << *params;
    int i = 0;
    int experiment;
    for (params->initLHS(params->N), experiment = 1; !params->finished(); params->setNextValues(), experiment++) {
        if (i >= 0 && i < 17) {

            file1 << params->a_t << "\n";
            file1 << params->lp << "\n";
            file1 << params->dsbeta << "\n";
            file1 << params->e_b << "\n";
            file1 << params->e_n << "\n";
            file1 << params->epsilon_ds << "\n";
            file1 << params->minwd << "\n";

        } else if (i >= 17 && i < 34) {

            file2 << params->a_t << "\n";
            file2 << params->lp << "\n";
            file2 << params->dsbeta << "\n";
            file2 << params->e_b << "\n";
            file2 << params->e_n << "\n";
            file2 << params->epsilon_ds << "\n";
            file2 << params->minwd << "\n";

        } else if (i >= 34 && i < 51) {

            file3 << params->a_t << "\n";
            file3 << params->lp << "\n";
            file3 << params->dsbeta << "\n";
            file3 << params->e_b << "\n";
            file3 << params->e_n << "\n";
            file3 << params->epsilon_ds << "\n";
            file3 << params->minwd << "\n";

        } else if (i >= 51 && i < 68) {

            file4 << params->a_t << "\n";
            file4 << params->lp << "\n";
            file4 << params->dsbeta << "\n";
            file4 << params->e_b << "\n";
            file4 << params->e_n << "\n";
            file4 << params->epsilon_ds << "\n";
            file4 << params->minwd << "\n";

        } else if (i >= 68 && i < 85) {

            file5 << params->a_t << "\n";
            file5 << params->lp << "\n";
            file5 << params->dsbeta << "\n";
            file5 << params->e_b << "\n";
            file5 << params->e_n << "\n";
            file5 << params->epsilon_ds << "\n";
            file5 << params->minwd << "\n";

        } else if (i >= 85 && i < 100) {

            file6 << params->a_t << "\n";
            file6 << params->lp << "\n";
            file6 << params->dsbeta << "\n";
            file6 << params->e_b << "\n";
            file6 << params->e_n << "\n";
            file6 << params->epsilon_ds << "\n";
            file6 << params->minwd << "\n";

        }
        i++;

    }
    file1.close();
    file2.close();
    file3.close();
    file4.close();
    file5.close();
    file6.close();
}

void createParametersFile(MyParameters * params) {
    string name1 = "/home/raphael/Desktop/MyParametersFile.txt";
    std::ofstream file1;
    file1.open(name1.c_str());
    cout << *params;
    int experiment;
    for (params->initLHS(params->N), experiment = 1; !params->finished(); params->setNextValues(), experiment++) {

        file1 << params->a_t << "\n";
        file1 << params->lp << "\n";
        file1 << params->dsbeta << "\n";
        file1 << std::round(params->age_wins) << "\n";
        file1 << params->e_b << "\n";
        file1 << params->e_n << "\n";
        file1 << params->epsilon_ds << "\n";
        file1 << params->minwd << "\n";

    }
    file1.close();
}

void loadFalseDataInMemory() {
    for (int inputSize = 2; inputSize <= 6; ++inputSize) {
        for (int fileNumber = 0; fileNumber < 100; ++fileNumber) {
            v_false[fileNumber].push_back(loadFalseData(inputSize, fileNumber));
        }
    }
}

void loadTrueDataInMemory() {
    for (int inputSize = 2; inputSize <= 6; ++inputSize) {
        for (int fileNumber = 0; fileNumber < 100; ++fileNumber) {
            v_true[fileNumber].push_back(loadTrueData(inputSize, fileNumber));
        }
    }
}

void runTimeSeriesMotifDiscovery(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, int paramsNumber, std::string &featuresDict, OutputMetrics outputM) {
    dbgOut(1) << "run Time Series Motif Discovery " << endl;

    som->reset();
    som->a_t = 0.4335352;
    som->dsbeta = 0.0922593;
    som->e_b = 0.0595755;
    som->e_n = 0.247219;
    som->epsilon_ds = 0.0698139;
    som->minwd = 0.222785;
    som->d_max = 2;


    //Faz o treinamento e teste para a quantidade de dimensões solicitadas com taxas

    for (int i = 150; i <= 150; i++) {
        //Taxa de true positive
        //createOneInputDataFromTimeSeries(i, "/home/raphael/git/pbml/Datasets/Gun_Point/TRAIN/Gun_Point_TRAIN");
        MatMatrix<float> data = loadDataFromPath("/home/raphael/git/pbml/Datasets/Gun_Point/Gun_Point_TRAIN_teste");
        clusteringSOM.setData(data);
        som->resetSize(clusteringSOM.getInputSize());
        clusteringSOM.trainSOM(1); // 1 - Epocs
        //som->saveSOM("networks/TimeSeries/som_arq_TE_" + std::to_string(i));
        dbgOut(1) << "som->size() = " << std::to_string(som->size()) << " D = " << std::to_string(i) << endl;
    }
    som->saveSOM("networks/TimeSeries/som_arq_0");
}

vector<string> splitString(string str, char delimiter) {
    vector<string> internal;
    stringstream ss(str); // Turn the string into a stream.
    string tok;

    while (getline(ss, tok, delimiter)) {
        internal.push_back(tok);
    }

    return internal;
}

void createInputDataFromTimeSeries(int dimension, string path) {
    std::ifstream file(path);
    string text;
    FeaturesVector featuresVector;
    std::vector<FeaturesVector> vectorData;
    MatMatrix<float> data;
    std::vector<int> groups;
    std::map<int, int> groupLabels;
    while (!file.eof()) {

        getline(file, text);
        vector<string> tempVector = splitString(text, ',');
        if (tempVector.size() < 1) {
            dbgOut(1) << text;
        }
        for (int i = 1; i < tempVector.size(); i++) {
            Features features;
            float number = std::stof(tempVector[i]);
            features.append(number);
            featuresVector.concatCols(features);
        }


    }
    vectorData.push_back(featuresVector);
    createInputData(vectorData, dimension, data, groupLabels, groups);
    string name_true = "/home/raphael/git/pbml/Datasets/Gun_Point/TRAIN/D-" + std::to_string(dimension) + "_arq_train.txt";
    std::ofstream file_out;
    file_out.open(name_true.c_str());

    for (int i = 0; i < data.rows(); i++) {
        for (int j = 0; j < data.cols(); j++) {
            file_out << data[i][j] << "\t";
        }
        file_out << "\n";
    }
    file_out.close();
}

MatMatrix<float> createOneInputDataFromTimeSeries(int dimension, string path) {
    std::ifstream file(path);
    string text;
    FeaturesVector featuresVector;
    std::vector<FeaturesVector> vectorData;
    MatMatrix<float> data;
    std::vector<int> groups;
    std::map<int, int> groupLabels;
    while (!file.eof()) {

        getline(file, text);
        vector<string> tempVector = splitString(text, ',');
        if (tempVector.size() < 1) {
            dbgOut(1) << text;
        }
        for (int i = 1; i < tempVector.size(); i++) {
            Features features;
            float number = std::stof(tempVector[i]);
            features.append(number);
            featuresVector.concatCols(features);
        }


    }
    vectorData.push_back(featuresVector);
    createInputData(vectorData, dimension, data, groupLabels, groups);
    return data;
}

void runTestAfterTrainingTimeSeries(VILMAP *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, int paramsNumber, std::string &featuresDict, OutputMetrics outputM) {
    string testname = "sentences.txt";
    som->d_max = 150;
    som->d_min = 150;

    MatMatrix<int> taxaTrue, taxaFalse; // 0 - Ativaçoes totais // 1 - Ativações reconhecidas // 2 - Ativações Não reconhecidas

    //Testa com todos os arquivos de entrada depois que arede já foi treinada
    som->readSOM("/home/raphael/git/pbml/visom/networks/TimeSeries/som_teste_D-2--3750");


    for (int i = som->d_min; i <= som->d_max; i++) { // For para tamanhos de entrada

        //Taxa de true positive
        MatMatrix<float> data = createOneInputDataFromTimeSeries(i, "/home/raphael/git/pbml/Datasets/Gun_Point/TRAIN/Gun_Point_TRAIN");
        clusteringSOM.setData(data);
        som->resetSize(clusteringSOM.getInputSize());
        taxaTrue.concatRows(clusteringSOM.writeClusterResultsReadable("output/result_" + std::to_string(i) + "_" + testname, data, featuresDict, dssom, som->a_t));

        //Taxa de false negative
        //                MatMatrix<float> dataFalse = loadFalseData(i, fileNumber);
        //                clusteringSOM.setData(dataFalse);
        //                som->resetSize(clusteringSOM.getInputSize());
        //                taxaFalse.concatRows(clusteringSOM.writeClusterResultsReadable("output/false_" + std::to_string(i) + "_" + testname, dataFalse, featuresDict, dssom, som->a_t));

    }

    //            outputM.PATH = "output" + std::to_string(paramsNumber + 10) + "/";
    //            outputM.outputWithParamsFiles(som, experiment, taxaTrue, taxaFalse, fileNumber);
    dbgOut(1) << std::to_string(100) << "% Concluido do arquivo " << endl;

}
