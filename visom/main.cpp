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
#include "Defines.h"
#include "DebugOut.h"
#include "Parameters.h"
#include "ClusteringMetrics.h"
#include "ArffData.h"
#include "ClusteringSOM.h"
#include "LARFDSSOM_REC.h"
#include "unistd.h"
#include "TextToPhoneme.h"
#include "unistd.h"
#include "LHSParameters.h"
#include "MyParameters.h"
#include <string>

using namespace std;

int loadFeatureMatrix(std::ifstream &file, std::string &dictionary, std::string &featuresDict, std::vector<FeaturesVector> &data, const std::string &filename) {
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

MatMatrix<float> loadFalseData(int tam, int cont) {
    MatMatrix<float> mat;
    std::ifstream inputFile("input/falseData_" + std::to_string(tam) + "_arq_" + std::to_string(cont));
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

MatMatrix<float> loadTrueData(int tam, int cont) {
    MatMatrix<float> mat;
    std::ifstream inputFile("input/trueData_" + std::to_string(tam) + "_arq_" + std::to_string(cont));
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

int main(int argc, char** argv) {

    //Teste de Parâmetros
    MyParameters params;

    int seed = 0;
    dbgThreshold(1);
    LARFDSSOM_REC som(1);
    SOM<DSNode> *dssom = (SOM<DSNode>*) & som;
    ClusteringMeshSOM clusteringSOM(dssom);

    som.maxNodeNumber = 20;
    som.e_b = 0.0005;
    som.e_n = 0.000001;
    //
    som.dsbeta = 0.001;
    som.epsilon_ds = 0.0;
    som.minwd = 0.5;
    som.age_wins = 10;
    som.lp = 0.06; //0.175;
    som.a_t = 0.95; //Limiar de atividade .95
    int epocs = 10000;
    clusteringSOM.setFilterNoise(true);
    clusteringSOM.setIsSubspaceClustering(true);
    string filename = "";
    //Parâmetros de dimensionalidade
    som.d_max = 6;
    som.d_min = 2;
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

        string testname = "sentences.txt";
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


    string dictionary = "dictionary/cmu-dict.csv";
    string featuresDict = "dictionary/PhonemeFeatures.csv";
    /*
        //Gerar arquivos de treinamento e teste
        std::vector<FeaturesVector> phonemesData;
        std::vector<FeaturesVector> phonemesFalseData;
        int phonemesNum = loadFeatureMatrix(file, dictionary, featuresDict, phonemesData, "phonemes_" + filename);
        loadFalseFeatureMatrix(featuresDict, phonemesFalseData, "phonemes_falseFile", phonemesNum);
        std::vector<int> groups;
        std::map<int, int> groupLabels;

        som.d_max = 6;
        for (int i = som.d_min; i <= som.d_max; i++) {
            string name_true = "trueData_" + std::to_string(i) + "_arq_" + std::to_string(cont);
            std::ofstream file_true;
            file_true.open(name_true.c_str());

            string name_false = "falseData_" + std::to_string(i) + "_arq_" + std::to_string(cont);
            std::ofstream file_false;
            file_false.open(name_false.c_str());

            //Taxa de true positive
            MatMatrix<float> data;
            createInputData(phonemesData, i, data, groupLabels, groups);
            for (int i = 0; i < data.rows(); i++) {
                for (int j = 0; j < data.cols(); j++) {
                    file_true << data[i][j] << "\t";
                }
                file_true << "\n";
            }

            //Taxa de false negative
            MatMatrix<float> dataFalse;
            createInputData(phonemesFalseData, i, dataFalse, groupLabels, groups);
            for (int i = 0; i < dataFalse.rows(); i++) {
                for (int j = 0; j < dataFalse.cols(); j++) {
                    file_false << dataFalse[i][j] << "\t";
                }
                file_false << "\n";
            }
        }
     */
    for (int tam = 0; tam < 100; tam++) {
        MatMatrix<int> taxaTrue, taxaFalse; // 0 - Ativaçoes totais // 1 - Ativações reconhecidas // 2 - Ativações Não reconhecidas
        int ciclo = tam+1;
        int cont = 0;
        //Faz o treinamento e teste para a quantidade de dimensões solicitadas com taxas
        som.d_max = 6;
        som.d_min = 2;
        for (int i = som.d_min; i <= som.d_max; i++) {
            som.readSOM("networks/som_0_" + std::to_string(ciclo) + "_TE_" + std::to_string(6));
            //Taxa de true positive
            MatMatrix<float> data = loadTrueData(i, cont);
            clusteringSOM.setData(data);
            //som.reset(clusteringSOM.getInputSize());
            //clusteringSOM.trainSOM(epocs);
            taxaTrue.concatRows(clusteringSOM.writeClusterResultsReadable("output/result_" + std::to_string(i) + "_" + testname, data, featuresDict, dssom, som.a_t));

            //Taxa de false negative
            MatMatrix<float> dataFalse = loadFalseData(i, cont);
            clusteringSOM.setData(dataFalse);
            taxaFalse.concatRows(clusteringSOM.writeClusterResultsReadable("output/false_" + std::to_string(i) + "_" + testname, dataFalse, featuresDict, dssom, som.a_t));
            //som.saveSOM("som_" + std::to_string(cont) + "_" + std::to_string(ciclo) + "_TE_" + std::to_string(i));
        }



        std::ofstream file1;
        std::string name = "output/metrics.txt";
        file1.open(name.c_str(), std::ios_base::app);

        file1 << "Params:" << endl;
        file1 << "a_t " << " = " << som.a_t << "  ";
        file1 << "dsbeta " << " = " << som.dsbeta << "  ";
        file1 << "e_b " << " = " << som.e_b << "  ";
        file1 << "e_n " << " = " << som.e_n << "  ";
        file1 << "epsilon_ds " << " = " << som.epsilon_ds << "  ";
        file1 << "lp " << " = " << som.lp << "  ";
        file1 << "minwd " << " = " << som.minwd << endl;
        file1 << "Experimento = " << ciclo;

        std::ofstream file2;
        std::string name2 = "output/metrics_read.txt";
        file2.open(name2.c_str(), std::ios_base::app);

        file2 << "\n\nParams:" << endl;
        file2 << "a_t " << " = " << som.a_t << "  ";
        file2 << "dsbeta " << " = " << som.dsbeta << "  ";
        file2 << "e_b " << " = " << som.e_b << "  ";
        file2 << "e_n " << " = " << som.e_n << "  ";
        file2 << "epsilon_ds " << " = " << som.epsilon_ds << "  ";
        file2 << "lp " << " = " << som.lp << "  ";
        file2 << "minwd " << " = " << som.minwd << endl;
        file2 << "Experimento = " << ciclo;
        /*
            //Arquivos por parâmetro
            std::ofstream file3;
            std::string name3 = params.a_t.name + ".txt";
            file3.open(name3.c_str(), std::ios_base::app);
            file3 << "---------------------- " << params.a_t.name << " = " << som.a_t << " ----------------------" << endl;
            file3 << "Arquivo = " << cont << " | Experimento = " << ciclo << endl;

            std::ofstream file4;
            std::string name4 = params.dsbeta.name + ".txt";
            file4.open(name4.c_str(), std::ios_base::app);
            file4 << "---------------------- " << params.dsbeta.name << " = " << som.a_t << " ----------------------" << endl;
            file4 << "Arquivo = " << cont << " | Experimento = " << ciclo << endl;

            std::ofstream file5;
            std::string name5 = params.e_b.name + ".txt";
            file5.open(name5.c_str(), std::ios_base::app);
            file5 << "---------------------- " << params.e_b.name << " = " << som.a_t << " ----------------------" << endl;
            file5 << "Arquivo = " << cont << " | Experimento = " << ciclo << endl;

            std::ofstream file6;
            std::string name6 = params.e_n.name + ".txt";
            file6.open(name6.c_str(), std::ios_base::app);
            file6 << "---------------------- " << params.e_n.name << " = " << som.a_t << " ----------------------" << endl;
            file6 << "Arquivo = " << cont << " | Experimento = " << ciclo << endl;

            std::ofstream file7;
            std::string name7 = params.epsilon_ds.name + ".txt";
            file7.open(name7.c_str(), std::ios_base::app);
            file7 << "---------------------- " << params.epsilon_ds.name << " = " << som.a_t << " ----------------------" << endl;
            file7 << "Arquivo = " << cont << " | Experimento = " << ciclo << endl;

            std::ofstream file8;
            std::string name8 = params.lp.name + ".txt";
            file8.open(name8.c_str(), std::ios_base::app);
            file8 << "---------------------- " << params.lp.name << " = " << som.a_t << " ----------------------" << endl;
            file8 << "Arquivo = " << cont << " | Experimento = " << ciclo << endl;

            std::ofstream file9;
            std::string name9 = params.minwd.name + ".txt";
            file9.open(name9.c_str(), std::ios_base::app);
            file9 << "---------------------- " << params.minwd.name << " = " << som.a_t << " ----------------------" << endl;
            file9 << "Arquivo = " << cont << " | Experimento = " << ciclo << endl;

         */
        for (int row = 0, d = som.d_min; row < taxaTrue.rows(); row++, d++) {

            file1 << "\nTamanho_da_entrada " << d << endl;
            file2 << "\nTamanho_da_entrada " << d << endl;
            /*
            file3 << "Tamanho_da_entrada " << d << endl;
            file4 << "Tamanho_da_entrada " << d << endl;
            file5 << "Tamanho_da_entrada " << d << endl;
            file6 << "Tamanho_da_entrada " << d << endl;
            file7 << "Tamanho_da_entrada " << d << endl;
            file8 << "Tamanho_da_entrada " << d << endl;
            file9 << "Tamanho_da_entrada " << d << endl;
             */

            file1 << "\nTrue Data" << endl;
            file1 << "Total de ativacoes = " << taxaTrue[row][0] << endl;
            file1 << "Verdadeiros positivos = " << taxaTrue[row][1] << endl;
            file1 << "Falsos negativos = " << taxaTrue[row][2] << endl;

            file1 << "\nFalse Data" << endl;
            file1 << "Total de Ativacoes = " << taxaFalse[row][0] << endl;
            file1 << "Falsos positivos = " << taxaFalse[row][1] << endl;
            file1 << "Verdadeiros negativos = " << taxaFalse[row][2] << endl;

            file1 << "\nMetricas" << endl;
            file1 << "Precision = ";
            float precision = (taxaTrue[row][1]) / (taxaFalse[row][1] + taxaTrue[row][1] + 0.000000000001);
            file1 << precision << endl;
            float recall = (taxaTrue[row][1]) / (taxaTrue[row][2] + taxaTrue[row][1] + 0.000000000001);
            file1 << "Recall = ";
            file1 << recall << endl;
            file1 << "F-measure = ";
            file1 << (2 * precision * recall) / (precision + recall + 0.0000000001) << endl;

            file2 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001);
            /*
            file3 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
            file4 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
            file5 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
            file6 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
            file7 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
            file8 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
            file9 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
             */

        }
        dbgOut(1) << std::to_string(tam) << "% Concluido do arquivo " << cont << endl;
        dbgOut(1) << "Done." << endl;
    }
}