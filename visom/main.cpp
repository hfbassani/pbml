/*
 * File:   main.cpp
 * Author: hans
 *
 * Created on 11 de Outubro de 2010, 07:25
 */
#define NO_INVALID_DIMENSION_SIZE

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
#include "VILARFDSSOM.h"
#include "unistd.h"
#include "TextToPhoneme.h"
#include "MyParameters/MyParameters.h"
#include "OutputMetrics/OutputMetrics.h"
#include <string>

using namespace std;

void createTrainingTestFiles(int d_min, int d_max, std::ifstream &file, std::string &dictionary, std::string &featuresDict, const std::string &filename);

int loadTrueFeatureMatrix(std::ifstream &file, std::string &dictionary, std::string &featuresDict, std::vector<FeaturesVector> &data, const std::string &filename);

void loadFalseFeatureMatrix(std::string &featuresDict, std::vector<FeaturesVector> &data, const std::string &filename, int phonemesNum);

void createInputData(std::vector<FeaturesVector> &data, int inputCount, MatMatrix<float> &mdata, std::map<int, int> &groupLabels, std::vector<int> &groups);

void createPhonemaData(std::string &featuresDict, MatMatrix<float> &data);

MatMatrix<float> loadFalseData(int tam, int fileNumber);

MatMatrix<float> loadTrueData(int tam, int fileNumber);

void runCompleteTest(VILARFDSSOM *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, MyParameters params, std::string &dictionary, std::string &featuresDict, OutputMetrics outputM);

void runTestAfterTraining(VILARFDSSOM *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, MyParameters params, std::string &dictionary, std::string &featuresDict, OutputMetrics outputM);

int main(int argc, char** argv) {

    MyParameters params; //Teste de Parâmetros
    OutputMetrics outputM; //Classe que escreve a saída das métricas
    int seed = 0;
    dbgThreshold(1);
    VILARFDSSOM som(1);
    SOM<DSNode> *dssom = (SOM<DSNode>*) & som;
    ClusteringMeshSOM clusteringSOM(dssom);

    string dictionary = "dictionary/cmu-dict.csv";
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
    clusteringSOM.setFilterNoise(true);
    clusteringSOM.setIsSubspaceClustering(true);
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


    if (filename == "") {
        dbgOut(0) << "option -f [filename] is required" << endl;
        return -1;
    }

    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        dbgOut(0) << "Error openning file: " << filename << endl;
        return false;
    }

    dbgOut(1) << "Running VILARFDSSOM for file: " << filename << endl;


    srand(time(0));
    srand(seed);



    //createTrainingTestFiles(som.d_min, som.d_max, file, dictionary, featuresDict, "phonemes_" + filename);
    runCompleteTest(&som, clusteringSOM, dssom, params, dictionary, featuresDict, outputM);
    //runTestAfterTraining(&som, clusteringSOM, dssom, params, dictionary, featuresDict, outputM);

    dbgOut(1) << "Done." << endl;
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
                output.append(text + " -> " + phonemes);
                pf.translatePhonemesFeatures(phonemes, features);
                data.push_back(features);
                phonemesNum += features.size();
            } else
                dbgOut(0) << "Unknown word: " << text << endl;
        }
    }

    for (int i = 0; i < output.size(); i++) {
        fileOutput << output[i];
        fileOutput << endl;
    }

    fileOutput.close();

    dbgOut(0) << phonemesNum / 12 << endl;
    return phonemesNum / 12;
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
    std::ifstream inputFile("input/falseData_" + std::to_string(tam) + "_arq_" + std::to_string(fileNumber));
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
    std::ifstream inputFile("input/trueData_" + std::to_string(tam) + "_arq_" + std::to_string(fileNumber));
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

void runCompleteTest(VILARFDSSOM *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, MyParameters params, std::string &dictionary, std::string &featuresDict, OutputMetrics outputM) {

    cout << params;
    int experiment = 1;
    int fileNumber = 0;
    string filename = "";
    for (params.initLHS(params.N), experiment = 1; !params.finished(); params.setNextValues(), experiment++) {
        som->a_t = params.a_t;
        som->lp = params.lp;
        som->dsbeta = params.dsbeta;
        som->e_b = params.e_b;
        som->e_n = params.e_n;
        som->epsilon_ds = params.epsilon_ds;
        som->minwd = params.minwd;
        for (fileNumber = 0; fileNumber < 100; fileNumber++) {
            filename = "input/sentences_" + std::to_string(fileNumber) + ".txt";

            MatMatrix<int> taxaTrue, taxaFalse; // 0 - Ativaçoes totais // 1 - Ativações reconhecidas // 2 - Ativações Não reconhecidas

            //Faz o treinamento e teste para a quantidade de dimensões solicitadas com taxas
            som->d_max = 6;
            for (int i = som->d_min; i <= som->d_max; i++) {
                //Taxa de true positive
                MatMatrix<float> data = loadTrueData(i, fileNumber);
                clusteringSOM.setData(data);
                som->reset(clusteringSOM.getInputSize());
                clusteringSOM.trainSOM(1); // 1 - Epocs
                taxaTrue.concatRows(clusteringSOM.writeClusterResultsReadable("output/result_" + std::to_string(i) + "_" + filename, data, featuresDict, dssom, som->a_t));

                //Taxa de false negative
                MatMatrix<float> dataFalse = loadFalseData(i, fileNumber);
                clusteringSOM.setData(dataFalse);
                taxaFalse.concatRows(clusteringSOM.writeClusterResultsReadable("output/false_" + std::to_string(i) + "_" + filename, dataFalse, featuresDict, dssom, som->a_t));
                som->saveSOM("networks/som_arq_" + std::to_string(fileNumber) + "_exp_" + std::to_string(experiment) + "_TE_" + std::to_string(i));
            }


            outputM.outputWithParamsFiles(som, experiment, taxaTrue, taxaFalse, params, fileNumber);

        }
    }
}

void runTestAfterTraining(VILARFDSSOM *som, ClusteringMeshSOM clusteringSOM, SOM<DSNode> *dssom, MyParameters params, std::string &dictionary, std::string &featuresDict, OutputMetrics outputM) {
    string testname = "sentences.txt";
    som->d_max = 6;
    som->d_min = 2;
    int fileNumber = 0;
    for (fileNumber = 0; fileNumber <= 100; fileNumber++) {
        MatMatrix<int> taxaTrue, taxaFalse; // 0 - Ativaçoes totais // 1 - Ativações reconhecidas // 2 - Ativações Não reconhecidas
        int experiment = fileNumber + 1;

        //Faz o treinamento e teste para a quantidade de dimensões solicitadas com taxas
        som->readSOM("networks/som_arq_" + std::to_string(fileNumber) + "_exp_" + std::to_string(experiment) + "_TE_" + std::to_string(6));
        for (int i = som->d_min; i <= som->d_max; i++) {
            
            //Taxa de true positive
            MatMatrix<float> data = loadTrueData(i, fileNumber);
            clusteringSOM.setData(data);
            taxaTrue.concatRows(clusteringSOM.writeClusterResultsReadable("output/result_" + std::to_string(i) + "_" + testname, data, featuresDict, dssom, som->a_t));

            //Taxa de false negative
            MatMatrix<float> dataFalse = loadFalseData(i, fileNumber);
            clusteringSOM.setData(dataFalse);
            taxaFalse.concatRows(clusteringSOM.writeClusterResultsReadable("output/false_" + std::to_string(i) + "_" + testname, dataFalse, featuresDict, dssom, som->a_t));
            
        }

        outputM.output(som, experiment, taxaTrue, taxaFalse);


        dbgOut(1) << std::to_string(fileNumber) << "% Concluido do arquivo " << fileNumber << endl;

    }
}