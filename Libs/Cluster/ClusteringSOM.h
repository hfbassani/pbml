/* 
 * File:   SOMType.h
 * Author: hans
 *
 * Created on 30 de Março de 2012, 13:15
 */

#ifndef CLUSTERINGSOM_H
#define CLUSTERINGSOM_H

#include <vector>
#include <map>
#include <iomanip>
#include <iostream>
#include "ArffData.h"
#include "MatVector.h"

#include "SOM.h"
#include "DSNode.h"
#include "ClusteringMetrics.h"
#include "TextToPhoneme.h"
#include <math.h>
#include <sstream>
#include <cstring>

template <class SOMType>
class ClusteringSOM {
public:
    SOMType *som;
    MatMatrix<float> *trainingData;
    bool allocated;
    std::vector<int> groups;
    std::map<int, int> groupLabels;

    bool isSubspaceClustering;
    bool filterNoise;

public:

    ClusteringSOM() {
        isSubspaceClustering = true;
        trainingData = NULL;
        allocated = false;
        filterNoise = true;
    };

    ClusteringSOM(SOMType *som) {
        this->som = som;
        isSubspaceClustering = true;
        trainingData = NULL;
        allocated = false;
        filterNoise = true;
    };

    virtual ~ClusteringSOM() {
        if (allocated)
            delete trainingData;
    };

    void setFilterNoise(bool filter) {
        filterNoise = filter;
    }

    void setIsSubspaceClustering(bool isSubspaceClustering) {
        this->isSubspaceClustering = isSubspaceClustering;
    }

    virtual int getMeshSize() = 0;

    int getInputSize() {
        return trainingData->cols();
    }

    int getNumSamples() {
        return trainingData->rows();
    }

    virtual int getNodeId(int node_i) {
        return node_i;
    }

    virtual void train(MatMatrix<float> &trainingData, int N) = 0;

    virtual void getRelevances(int node_i, MatVector<float> &relevances) = 0;

    virtual void getWeights(int node_i, MatVector<float> &weights) = 0;

    virtual void getWinners(const MatVector<float> &sample, std::vector<int> &winners) = 0;

    virtual int getWinner(const MatVector<float> &sample) = 0;

    virtual bool isNoise(const MatVector<float> &sample) {
        return false;
    }

    void cleanUpTrainingData() {
        trainingData->clear();
        groups.clear();
        groupLabels.clear();
    }

    bool readFile(const std::string &filename) {

        if (trainingData == NULL && !allocated) {
            trainingData = new MatMatrix<float>();
            allocated = true;
        }

        if (ArffData::readArff(filename, *trainingData, groupLabels, groups)) {
            ArffData::rescaleCols01(*trainingData);
            return true;
        }
        return false;
    }

    void setData(MatMatrix<float> &data) {
        if (allocated) {
            delete trainingData;
            allocated = false;
        }

        trainingData = &data;
    }

    void trainSOM(int N) {

        train(*trainingData, N);
    }

    bool writeClusterResults(const std::string &filename) {

        std::ofstream file;
        file.open(filename.c_str());

        if (!file.is_open()) {
            dbgOut(0) << "Error openning output file" << endl;
            return false;
        }

        int meshSize = getMeshSize();
        int inputSize = getInputSize();

        file << meshSize << "\t" << inputSize << endl;

        for (int i = 0; i < meshSize; i++) {
            MatVector<float> relevances;
            getRelevances(i, relevances);

            file << i << "\t";
            for (int j = 0; j < inputSize; j++) {
                file << relevances[j];
                if (j != inputSize - 1)
                    file << "\t";
            }
            file << endl;
        }

        for (int i = 0; i < trainingData->rows(); i++) {
            MatVector<float> sample;
            trainingData->getRow(i, sample);
            if (filterNoise && isNoise(sample))
                continue;

            std::vector<int> winners;
            if (isSubspaceClustering) {
                getWinners(sample, winners);
            } else {
                winners.push_back(getWinner(sample));
            }

            for (int j = 0; j < winners.size(); j++) {
                file << i << "\t";
                file << winners[j];
                file << endl;
            }
        }

        file.close();
        return true;
    }

    float activation(DSNode* node, MatVector<float> &w) {
        int end = 0;
        float tempDistance = 0;
        node->index = 0;
        float distance = 0;
        if (node->w.size() <= w.size()) {
            end = node->w.size();
            for (uint i = 0; i < end; i++) {
                distance += node->ds[i] * qrt((w[i] - node->w[i]));
                if (std::isnan(w[i]) || std::isnan(distance)) {
                    std::cout << i << " - Debug 1" << endl;
                }
            }
        } else {

            distance = 999;
            for (uint i = 0; i <= (node->w.size() - w.size()); i += 12) {
                tempDistance = 0;
                for (uint j = 0; j < w.size(); j++) {
                    //cout << i << " - " << j << " - " << tempDistance << " | ";
                    tempDistance += node->ds[i + j] * qrt((w[j] - node->w[i + j]));
                    if (std::isnan(w[j]) || std::isnan(tempDistance)) {
                        std::cout << i << " - Debug 2" << endl;
                    }

                }

                if (tempDistance < distance) {
                    distance = tempDistance;
                    node->index = i;
                }
            }
        }


        //dbgOut(1) <<"N:" << node.w.size() << "\t" << "E:" << w.size();


        float sum = node->ds.sum();

        return (sum / (sum + distance + 0.0000001));

    }

    MatVector<int> writeClusterResultsReadable(const std::string &filename, MatMatrix<float> &data, std::string &featuresDict,
            SOM<DSNode> *som, float at_min) {

#ifdef PRINT_CLUSTER
        DSNode* nodoReset; //For reset all actvations
        nodoReset = som->getFirstNode();
        for (int i = 0; i < som->size(); i++) {
            nodoReset->at_Know = 0;
            nodoReset->at_Unknown = 0;
            if (i < som->size() - 1) {
                nodoReset = som->getNextNode();
            }
        }

        std::ofstream file;
        file.open(filename.c_str());
#endif
        int somTam = som->size();
        float a;
        int at_know = 0, at_all = 0, at_Unknown = 0;
        MatVector<int> activations;
        PhonemesToFeatures pf;
        std::string phonemes;
        pf.loadPhonemeFeatures(featuresDict, 12);
        Features features(12);
        MatVector<float> colOfData;
        data.getCol(0, colOfData);
#ifdef PRINT_CLUSTER
        if (!file.is_open()) {
            dbgOut(0) << "Error openning output file" << endl;
            return activations;
        }
#endif
        //Salva todos os protótipos da rede em um MatVector
        DSNode* nodoNow;
        MatVector<std::string> output_indice;
        std::vector<int> indice;
        MatVector<std::string> output_prototype;
        std::string temp2;

        nodoNow = som->getFirstNode();
        for (int i = 0; i < somTam; i++) {

            for (int begin = 0, end = 11, j = 0; end < nodoNow->w.size(); begin += 12, end += 12, j++) {
                features.copy(nodoNow->w, begin, end);
                pf.translateFeaturesPhoneme(features, phonemes);
                if (begin == 0) {
                    temp2 = phonemes;
                } else {
                    temp2 += " " + phonemes;
                }

            }
            /*/GAMBI INIT
            temp2 = nodoNow->w.toString();


            //GAMBI END*/

            MatVector<float> relevances;
            relevances = nodoNow->ds;
            std::string strRelevances;
            float average = 0, averageX2 = 0, deviation = 0;
            for (int begin = 0, end = 11; end < relevances.size(); begin += 12, end += 12) {
                average = 0, averageX2 = 0, deviation = 0;
                for (int i = begin; i <= end; i++) {
                    average += relevances[i]*(1 / 12.0);
                    averageX2 += relevances[i] * relevances[i]*(1 / 12.0);
                }
                deviation = sqrt(averageX2 - (average * average));
                strRelevances += std::to_string(average) + " +|- " + std::to_string(deviation) + "\t";
            }
            //P - protótipo / R - Relevancias / -A Ativacoes do nodo
            temp2 = " -P: " + temp2 + " -R: " + strRelevances + "\n" + "-A:";
            output_indice.append(std::to_string(nodoNow->getId()));
            indice.insert(indice.end(), nodoNow->getId());
            output_prototype.append(temp2);
            temp2 = "";
            if (i < somTam - 1) {
                nodoNow = som->getNextNode();
            }
        }

        //Cria Matriz que vai ser impressa no arquivo colocando ao lado de cada protótipo os seus respectivos vencedores
        MatMatrix<std::string> output_matrix;
        output_matrix.concatCols(output_indice);
        output_matrix.concatCols(output_prototype);
        for (int i = 0; i < trainingData->rows(); i++) {
            MatVector<float> sample;
            trainingData->getRow(i, sample);
            if (filterNoise && isNoise(sample))
                continue;

            std::vector<int> winners;
            DSNode* winner = som->getWinner(sample);
            winners.push_back(winner->getId());
            //Verificar ativação para calculo de métricas 
            
            a = activation(winner, sample);

            if (a >= at_min) {
                at_know++;
                winner->at_Know = winner->at_Know + 1;
            } else {
                winner->at_Unknown = winner->at_Unknown + 1;
                at_Unknown++;
            }
            at_all++;
            std::string temp;
            MatVector<float> rowOfData;
            data.getRow(i, rowOfData);
            for (int begin = 0, end = 11, j = 0; end < rowOfData.size(); begin += 12, end += 12, j++) {
                features.copy(rowOfData, begin, end);
                pf.translateFeaturesPhoneme(features, phonemes);
                if (begin == 0) {
                    temp += phonemes;
                } else
                    temp += " " + phonemes;
            }
            for (int j = 0; j < winners.size(); j++) {
                int aux = winners[j];
                int indice = 0, atual;
                for (int cont = 0; cont < output_matrix.rows(); cont++) {
                    atual = atoi(output_matrix[cont][0].c_str());
                    if (atual == aux) {
                        indice = cont;
                        break;
                    }

                }

                output_matrix[indice][1] += "  " + temp;



            }

        }
        //Salvar taxas
        activations.append(at_all);
        activations.append(at_know);
        activations.append(at_Unknown);
#ifdef PRINT_CLUSTER
        //Ordena a matriz de saída -> InsertionSort
        int i, j;
        std::string aux, aux1;
        for (i = 0; i < output_matrix.rows(); i++) {
            j = i;

            while ((j != 0) && (atoi(output_matrix[j][0].c_str()) < atoi(output_matrix[j - 1][0].c_str()))) {
                aux = output_matrix[j][0];
                aux1 = output_matrix[j][1];
                output_matrix[j][0] = output_matrix[j - 1][0];
                output_matrix[j][1] = output_matrix[j - 1][1];
                output_matrix[j - 1][0] = aux;
                output_matrix[j - 1][1] = aux1;
                j--;
            }
        }
        file << "Nodos = " << output_matrix.rows() << "\n";
        for (int i = 0; i < output_matrix.rows(); i++) {
            for (int j = 0; j < output_matrix.cols(); j++) {
                if (j == 0) {
                    file << "ID: " << output_matrix[i][j];
                    //Coloca taxas de ativacao de cada nodo no arquivo.
                    DSNode* nodo;
                    nodo = som->getFirstNode();
                    for (int cont = 0; cont < somTam; cont++) {
                        if (nodo->getId() == atoi(output_matrix[i][j].c_str())) {
                            file << " -AT_Know: " << std::to_string(nodo->at_Know) << " -AT_Unknow: " << std::to_string(nodo->at_Unknown) << " -G: " << std::to_string(nodo->generation);
                            break;
                        }

                        if (cont < somTam - 1) {
                            nodo = som->getNextNode();
                        }
                    }
                    //
                } else {
                    file << output_matrix[i][j] << "\n";
                }

            }
            file << endl;
        }

        dbgOut(1) << endl;

        file.close();
#endif
        dbgOut(1) << endl;
        return activations;
    }

    bool haveWord(MatVector<std::string> input, char* word) {
        char *res;

        for (int i = 0; i < input.size(); i++) {
            res = strstr(word, input[i].c_str());
            if (res != NULL) {
                return true;
            }
        }
        return false;

    }

    MatVector<int> writeClusterResultsArticle(const std::string &filename, MatMatrix<float> &data, std::string &featuresDict,
            SOM<DSNode> *som, float at_min) {
        std::ifstream inputFile("input/c1_cat_phonems.txt");
        std::string text;
        MatVector<std::string> dictOfWords;
#ifdef SAVE_NOT_WORDS
        MatVector<int> indiceNotWords;
#endif
        while (!inputFile.eof()) {
            getline(inputFile, text);
            dictOfWords.append(text);
        }
        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;
        int tem = 0;
        int nTem = 0;
        int somTam = som->size();
        float a;
        int at_know = 0, at_all = 0, at_Unknown = 0;
        MatVector<int> activations;
        PhonemesToFeatures pf;
        std::string phonemes;
        pf.loadPhonemeFeatures(featuresDict, 12);
        Features features(12);
        MatVector<float> colOfData;
        data.getCol(0, colOfData);

        //Cria Matriz que vai ser impressa no arquivo colocando ao lado de cada protótipo os seus respectivos vencedores

        for (int i = 0; i < trainingData->rows(); i++) {
            MatVector<float> sample;
            MatVector<float> sampleFilter;
            trainingData->getRow(i, sample);
            
            for(int t = 0; t < sample.size(); t++){
                if (sample[t] != 5){
                    sampleFilter.append(sample[t]);
                }else{
                    break;
                }
            }
            
            if (filterNoise && isNoise(sampleFilter))
                continue;

            std::vector<int> winners;
            DSNode* winner = som->getWinner(sampleFilter);
            winners.push_back(winner->getId());
            //Verificar ativação para calculo de métricas 
            
            a = activation(winner, sampleFilter);
            
            if (a >= at_min) {
                at_know++;
                winner->at_Know = winner->at_Know + 1;
            } else {
                winner->at_Unknown = winner->at_Unknown + 1;
                at_Unknown++;
            }
            at_all++;

            std::string temp;
            MatVector<float> rowOfDataFilter;
            MatVector<float> rowOfData;
            data.getRow(i, rowOfData);
            for(int t = 0; t < rowOfData.size(); t++){
                if (rowOfData[t] != 5){
                    rowOfDataFilter.append(rowOfData[t]);
                }else{
                    break;
                }
            }
            for (int begin = 0, end = 11, j = 0; end < rowOfDataFilter.size(); begin += 12, end += 12, j++) {
                features.copy(rowOfDataFilter, begin, end);
                pf.translateFeaturesPhoneme(features, phonemes);

                temp += phonemes;

            }


            if (haveWord(dictOfWords, (char *) temp.c_str())) {
                if (a >= at_min) {
                    tp++;
                } else {
                    fn++;
                }

            } else {
#ifdef SAVE_NOT_WORDS
                indiceNotWords.append(i);
#endif
                if (a >= at_min) {
                    fp++;
                } else {
                    tn++;
                }
                nTem++;
            }

        }
        //Salvar taxas
        activations.append(at_all);
        activations.append(at_know);
        activations.append(at_Unknown);
        activations.append(tp);
        activations.append(fp);
        activations.append(tn);
        activations.append(fn);
#ifdef SAVE_NOT_WORDS
        std::ofstream file_words;
        file_words.open("notWords.txt");
        for (int i = 0; i < indiceNotWords.size(); i++) {
            for (int j = 0; j < data.cols(); j++) {
                file_words << data[indiceNotWords[i]][j] << "\t";
            }
            file_words << "\n";
        }
#endif
        dbgOut(1) << endl;
        return activations;
    }

    int map(float x, float in_min, float in_max, float out_min, float out_max) {
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
    }

    void writeClusterResultsHTML(const std::string &filename, MatMatrix<float> &data, std::string &featuresDict,
            SOM<DSNode> *som, float at_min, int dim, int exp, int fileNumber) {
        using namespace std;

        MatVector<std::string> phonemes_now;
        MatVector<float> averages;
        std::vector<std::string> colors = {"#0000ff", "#0040ff", "#0080ff", "#00bfff", "#00ffff", "#00ffbf", "#00ff80", "#00ff40", "#00ff00", "#40ff00", "#80ff00", "#bfff00", "#ffff00", "#ffbf00", "#ff8000", "#ff4000", "#ff0000"};
        float a;
        int at_know = 0, at_all = 0, at_Unknown = 0;
        PhonemesToFeatures pf;
        std::string phonemes, phonemes_prototype;
        pf.loadPhonemeFeatures(featuresDict, 12);
        Features features(12);
        MatMatrix<std::string> output_matrix;

        ///HTML
        ofstream arq(filename);
        string buff = "<html>";
        arq.write(buff.c_str(), buff.length());
        buff = "<head><title>VILMAP - " + std::to_string(dim) + " </title></head>";
        arq.write(buff.c_str(), buff.length());
        buff = "<body bgcolor=#999999>";
        arq.write(buff.c_str(), buff.length());
        buff = "<center><h2>  Exp: " + std::to_string(exp) + " | Arq: " + std::to_string(fileNumber) + " | Tamanho de entrada = " + std::to_string(dim) + "</h2><center>";
        buff += "<center><h3>Frase: Base completa | Quantidade de nodos = " + std::to_string(som->size()) + "  | a_t = " + std::to_string(at_min) + "</h3><center>";
        buff += "<center><h3>Fonemas: Base completa  </h3><center>";
        arq.write(buff.c_str(), buff.length());
        buff = "<table border=1>";
        arq.write(buff.c_str(), buff.length());

        buff = "";

        /////HTML

        for (int i = 0; i < trainingData->rows(); i++) {
            MatVector<float> sample;
            trainingData->getRow(i, sample);
            if (filterNoise && isNoise(sample))
                continue;
            //cout << "W: " << i << endl;
            std::vector<int> winners;
            DSNode* winner = som->getWinnerCluster(sample);
            winners.push_back(winner->getId());
            //Verificar ativação para calculo de métricas 

            
            a = activation(winner, sample);
            uint index = winner->index;
            if (a >= at_min) {
                at_know++;
                winner->at_Know = winner->at_Know + 1;
            } else {
                winner->at_Unknown = winner->at_Unknown + 1;
                at_Unknown++;
            }
            at_all++;
            std::string temp;
            MatVector<float> rowOfData;
            data.getRow(i, rowOfData);
            for (int begin = 0, end = 11, j = 0; end < rowOfData.size(); begin += 12, end += 12, j++) {
                features.copy(rowOfData, begin, end);
                pf.translateFeaturesPhoneme(features, phonemes);
                if (begin == 0) {
                    temp += phonemes;
                } else {
                    temp += "&nbsp" + phonemes;
                }
                phonemes_now.append(phonemes);
            }
            MatVector<float> relevances;
            relevances = winner->ds;
            std::string strRelevances;
            float average = 0, averageX2 = 0, deviation = 0;
            for (int begin = 0, end = 11; end < relevances.size(); begin += 12, end += 12) {
                average = 0, averageX2 = 0, deviation = 0;
                for (int i = begin; i <= end; i++) {
                    average += relevances[i]*(1 / 12.0);
                    averageX2 += relevances[i] * relevances[i]*(1 / 12.0);
                }
                deviation = sqrt(averageX2 - (average * average));

                strRelevances += std::to_string(average) + " +|- " + std::to_string(deviation) + "\t";
                averages.append(average);
            }


            MatVector<int> indices;

            for (int x = 0; x < averages.size(); x++) {
                //                if (a < at_min) {
                //                    a = -a;
                //                }
                float v = averages[x] * (a); ///Fator de peso a_t interferindo na cor da relevância 
                if (v > 1) {
                    v = 1;
                } else if (v < 0) {
                    v = 0;
                }
                indices.append(map(v, 0.0, 1.0, 0, colors.size() - 1));
            }

            buff += "<tr>";

            buff += "<td>&nbsp;";
            //Coloca os fonemas da entrada na tabela
            for (int i = index / 12, j = 0; j < phonemes_now.size(); i++, j++) {
                buff += "<font color=" + colors[indices[i]] + "><b>";
                buff += phonemes_now[j];
                buff += "&nbsp;</b></font>";
            }

            buff += "</td>";
            //Vencedor
            buff += "<td>&nbsp;";
            buff += std::to_string(winner->getId());
            buff += "&nbsp;</td>";

            //Ativação
            float min = 0.7;
            if (min > a) {
                min = a;
            }
            int value = map(a, min, 1.0, 0, colors.size() - 1);
            //cout << endl << value << endl << " buff: " << a << endl << " - size: " << colors.size();

            buff += "<td>&nbsp;<font color=" + colors[value] + "><b>";

            stringstream stream;
            stream << fixed << setprecision(2) << a;
            buff += stream.str();
            buff += "&nbsp;</b></font></td>";
            ////Traduzir protótipos para fonemas
            phonemes_now.clear();
            for (int begin = 0, end = 11, j = 0; end < winner->w.size(); begin += 12, end += 12, j++) {
                features.copy(winner->w, begin, end);
                pf.translateFeaturesPhoneme(features, phonemes);
                phonemes_now.append(phonemes);

            }
            //Médias das relevancias por fonema
            for (int i = 0, j = 0; i < averages.size(); i++) {
                buff += "<td><center>&nbsp;";
                stringstream stream;
                stream << fixed << setprecision(2) << averages[i];
                buff += stream.str();
                if (i >= index / 12 && j < (sample.size() / 12)) {//<b> e </b>
                    buff += "<br><b>";
                    buff += phonemes_now[i];
                    buff += "&nbsp;</b></center></td>";
                    j++;
                } else {
                    buff += "<br>";
                    buff += phonemes_now[i];
                    buff += "&nbsp;</center></td>";
                }
            }

            buff += "</tr>";
            phonemes_now.clear();
            averages.clear();


        }



        arq.write(buff.c_str(), buff.length());

        buff = "</table>";
        arq.write(buff.c_str(), buff.length());
        buff = "</body>";
        arq.write(buff.c_str(), buff.length());
        buff = "</html>";
        arq.write(buff.c_str(), buff.length());

        arq.flush();
        arq.close();
        dbgOut(1) << endl;
    }

    std::string outClusters(bool printData = true) {

        std::stringstream out;
        out << std::setprecision(2) << std::fixed;

        int meshSize = getMeshSize();
        int inputSize = getInputSize();
        std::vector<int> clusterData;
        for (int i = 0; i < meshSize; i++) {

            out << getNodeId(i) << ": ";

            MatVector<float> relevances;
            getRelevances(i, relevances);

            //Print relevances
            float average = 0;
            int num = 0;
            for (int j = 0; j < inputSize; j++) {
                if (relevances[j] <= 1) {
                    average += relevances[j];
                    num++;
                }
            }
            if (num < 1) num = 1;
            average = average / num;

            for (int j = 0; j < inputSize; j++) {
                if (relevances[j] > average)
                    out << 1 << " ";
                else
                    out << 0 << " ";
            }

            clusterData.clear();
            for (int k = 0; k < trainingData->rows(); k++) {
                MatVector<float> sample;
                trainingData->getRow(k, sample);
                if (isNoise(sample))
                    continue;

                std::vector<int> winners;
                if (isSubspaceClustering) {
                    getWinners(sample, winners);
                } else {
                    winners.push_back(getWinner(sample));
                }

                for (int j = 0; j < winners.size(); j++) {
                    if (winners[j] == i) {
                        clusterData.push_back(k);
                        break;
                    }
                }
            }

            out << clusterData.size();
            if (printData) {
                out << "\t";
                for (int j = 0; j < clusterData.size(); j++) {
                    out << clusterData[j] << " ";
                }
            }
            out << endl;
        }

        return out.str();
    }

    std::string outRelevances() {

        std::stringstream out;
        out << std::setprecision(2) << std::fixed;

        int meshSize = getMeshSize();
        int inputSize = getInputSize();

        out << "D:\t";
        for (int j = 0; j < inputSize; j++) {
            out << j << "\t";
        }
        out << endl;

        for (int i = 0; i < meshSize; i++) {

            out << getNodeId(i) << ":\t";

            MatVector<float> relevances;
            getRelevances(i, relevances);

            //Print relevances
            for (int j = 0; j < inputSize; j++) {
                out << relevances[j] << "\t";
            }
            out << endl;
        }

        return out.str();
    }

    std::string outWeights() {

        std::stringstream out;

        out << std::setprecision(2) << std::fixed;

        int meshSize = getMeshSize();
        int inputSize = getInputSize();

        out << "D:\t";
        for (int j = 0; j < inputSize; j++) {
            out << j << "\t";
        }
        out << endl;

        for (int i = 0; i < meshSize; i++) {

            out << getNodeId(i) << ":\t";

            MatVector<float> weights;
            getWeights(i, weights);

            //Print relevances
            for (int j = 0; j < inputSize; j++) {
                out << weights[j] << "\t";
            }
            out << endl;
        }

        return out.str();
    }

    std::string outClassInfo() {

        std::stringstream out;
        out << std::setprecision(2) << std::fixed;

        int meshSize = getMeshSize();
        int hits = 0;
        int total = 0;
        int noise = 0;

        MatVector<int> nodeHits(meshSize);
        MatVector<int> nodeClusterSize(meshSize);
        nodeHits.fill(0);
        nodeClusterSize.fill(0);

        for (int k = 0; k < trainingData->rows(); k++) {
            MatVector<float> sample;
            trainingData->getRow(k, sample);
            if (isNoise(sample)) {
                noise++;
                continue;
            }

            int classIndex = sample.size() - 1;
            int winner = getWinner(sample);
            MatVector<float> weights;
            getWeights(winner, weights);

            if (fabs(sample[classIndex] - weights[classIndex]) < 0.5) {
                hits++;
                nodeHits[winner]++;
            }

            total++;
            nodeClusterSize[winner]++;
        }

        for (int i = 0; i < meshSize; i++) {

            MatVector<float> weights;
            getWeights(i, weights);

            out << getNodeId(i) << ": ";
            out << "\t" << weights[weights.size() - 1];
            out << "\t" << nodeHits[i] << "/" << nodeClusterSize[i];
            out << "\t" << nodeHits[i] / (float) nodeClusterSize[i];
            out << endl;
        }

        out << "Classification acuracy:\t" << hits / (float) total << endl;
        out << "Total noise:\t" << noise / (float) trainingData->rows() << endl;

        return out.str();
    }

    std::string outConfusionMatrix(std::vector<int> &groups, std::map<int, int> &groupLabels) {

        std::stringstream out;
        out << std::setprecision(2) << std::fixed;

        int meshSize = getMeshSize();
        MatMatrix<int> confusionMatrix(getMeshSize(), groupLabels.size());
        confusionMatrix.fill(0);
        int noise = 0;

        for (int k = 0; k < trainingData->rows(); k++) {
            MatVector<float> sample;
            trainingData->getRow(k, sample);
            if (isNoise(sample) && filterNoise) {
                noise++;
                continue;
            }

            std::vector<int> winners;
            if (isSubspaceClustering) {
                getWinners(sample, winners);
            } else {
                winners.push_back(getWinner(sample));
            }

            for (int w = 0; w < winners.size(); w++) {
                confusionMatrix[winners[w]][groups[k]]++;
            }
        }

        /** print confusion matrix **/
        MatVector<int> rowSums(confusionMatrix.rows());
        MatVector<int> colSums(confusionMatrix.cols());
        rowSums.fill(0);
        colSums.fill(0);
        dbgOut(0) << "cluster\\class\t|";
        for (int c = 0; c < confusionMatrix.cols(); c++)
            dbgOut(1) << "\tcla" << groupLabels[c];
        dbgOut(0) << "\t| Sum" << endl;
        for (int r = 0; r < confusionMatrix.rows(); r++) {
            dbgOut(0) << "clu" << r << "\t\t|";
            for (int c = 0; c < confusionMatrix.cols(); c++) {
                dbgOut(0) << "\t" << confusionMatrix[r][c];
                rowSums[r] += confusionMatrix[r][c];
                colSums[c] += confusionMatrix[r][c];
            }
            dbgOut(0) << "\t| " << rowSums[r] << endl;
        }
        dbgOut(0) << "Sums\t\t|";
        for (int c = 0; c < confusionMatrix.cols(); c++)
            dbgOut(0) << "\t" << colSums[c];
        dbgOut(0) << "\t| " << colSums.sum() << endl << endl;
        /***/

        dbgOut(0) << "Random index: " << ClusteringMetrics::RANDI(confusionMatrix) << endl;
        dbgOut(0) << "Adjusted random index: " << ClusteringMetrics::ARI(confusionMatrix) << endl << endl;
        out << "Total noise:\t" << noise << "(" << noise / (float) trainingData->rows() << ")" << endl;

        return out.str();
    }

    std::string printConditionalEntropy(std::vector<int> &groups) {
        std::stringstream out;
        MatVector<int> trueClusters(groups.size());
        MatVector<int> obtained(groups.size());
        int noise = 0;

        for (int k = 0; k < trainingData->rows(); k++) {
            MatVector<float> sample;
            trainingData->getRow(k, sample);

            //            if (isNoise(sample)) {
            //                noise++;
            //                continue;
            //            }

            int w = getWinner(sample);

            trueClusters[k] = groups[k];
            obtained[k] = w;
            //dbgOut(0) << trueClusters[k] << "\t" << obtained[k] << endl;

        }

        out << "Conditional entropy:\t" << conditionalEntropy(trueClusters, obtained) << endl;
        out << "Total noise:\t" << noise << "(" << noise / (float) trainingData->rows() << ")" << endl;
        return out.str();
    }

    float conditionalEntropy(MatVector<int> &X, MatVector<int> &Y) {
        std::map<int, int> m;

        for (int i = 0; i < Y.size(); i++) {
            int cluster = Y[i];

            if (m.find(cluster) != m.end()) {
                int count = m[cluster];
                m[cluster] = count + 1;
            } else {
                m[cluster] = 1;
            }
        }

        for (int i = 0; i < X.size(); i++) {
            int cluster = X[i];

            if (m.find(cluster) != m.end()) {
                int count = m[cluster];
                m[cluster] = count + 1;
            } else {
                m[cluster] = 1;
            }
        }

        int ny = m.size();

        MatVector<double> py(ny);
        MatMatrix<double> pxDy(ny, ny);
        py.fill(0);
        pxDy.fill(0);

        //Count y and (x,y) ocurrences
        for (int i = 0; i < Y.size(); i++) {
            pxDy[X[i]][Y[i]]++;
            py[Y[i]]++;
        }

        //Calculate p(x|y)
        for (int x = 0; x < ny; x++) {
            for (int y = 0; y < ny; y++) {
                if (py[y] > 0)
                    pxDy[x][y] /= py[y];
                else
                    pxDy[x][y] = 0;
            }
        }

        //Calculate p(y)
        for (int y = 0; y < ny; y++) {
            py[y] /= Y.size();
        }

        //dbgOut(0) << py.toString() << endl;
        //Compute conditional entropy: H(X|Y) = sum_y{py*sum_x[p(x|y)*log(1/p(x|y))]}
        double hxDy = 0;
        for (int y = 0; y < ny; y++) {

            float sum_pxDylog1_pxDy = 0;
            for (int x = 0; x < ny; x++) {
                if (pxDy[x][y] > 0)
                    sum_pxDylog1_pxDy += pxDy[x][y] * log2(1 / pxDy[x][y]);
            }
            hxDy += py[y] * sum_pxDylog1_pxDy;
        }

        return hxDy;
    }
};

#ifdef SOM2D_H

class ClusteringSOM2D : public ClusteringSOM<SOM2D<DSNeuron, SOMParameters> > {
public:

    ClusteringSOM2D(SOM2D<DSNeuron, SOMParameters> *som) : ClusteringSOM<SOM2D<DSNeuron, SOMParameters> >(som) {
    };

    int getMeshSize() {
        return som->getSomCols() * som->getSomRows();
    }

    void train(MatMatrix<float> &trainingData, int N) {
        som->parameters.NFeatures = trainingData.cols();
        som->parameters.tmax = N;
        som->initializeP(som->parameters);
        som->train(trainingData);
    }

    void getRelevances(int node_i, MatVector<float> &relevances) {
        relevances.size(som->parameters.NFeatures);
        relevances.fill(1);
    }

    void getWeights(int node_i, MatVector<float> &weights) {
        int col = node_i / som->getSomRows();
        int row = ((node_i / (float) som->getSomRows()) - col) * som->getSomRows();

        DSNeuron neuron(row, col);
        som->getNeuron(neuron);

        weights = neuron.weights;
    }

    int getWinner(const MatVector<float> &sample) {

        DSNeuron bmu;
        som->findBMU(sample, bmu);
        return bmu.c + bmu.r * som->getSomCols();
    }

    void getWinners(const MatVector<float> &sample, std::vector<int> &winner) {

        winner.clear();

        DSNeuron bmu;
        som->findBMU(sample, bmu);
        winner.push_back(bmu.c + bmu.r * som->getSomCols());
    }
};
#endif

#ifdef DSLVQ_H

class ClusteringDSLVQ : public ClusteringSOM<DSSOM<LVQNeuron, DSSOMParameters> > {
public:

    ClusteringDSLVQ(DSSOM<LVQNeuron, DSSOMParameters> *som) : ClusteringSOM<DSSOM<LVQNeuron, DSSOMParameters> >(som) {
    };

    int getMeshSize() {
        return som->getSomCols() * som->getSomRows();
    }

    void train(MatMatrix<float> &trainingData, int N) {
        som->parameters.NFeatures = trainingData.cols();
        som->parameters.tmax = N;
        som->initializeP(som->parameters);
        som->train(trainingData);
    }

    void test(MatMatrix<float> &testData, MatVector<int> classes) {

        classes.size(testData.rows());

        for (int r = 0; r < testData.rows(); r++) {
            MatVector<float> sample;
            testData.getRow(r, sample);

            LVQNeuron neuron;
            som->findBMU(sample, neuron);
            classes[r] = neuron.getClass();
        }
    }

    void getRelevances(int node_i, MatVector<float> &relevances) {
        int col = node_i / som->getSomRows();
        int row = ((node_i / (float) som->getSomRows()) - col) * som->getSomRows();

        LVQNeuron neuron(row, col);
        som->getNeuron(neuron);

        relevances = neuron.dsWeights;
    }

    void getWeights(int node_i, MatVector<float> &weights) {
        int col = node_i / som->getSomRows();
        int row = ((node_i / (float) som->getSomRows()) - col) * som->getSomRows();

        LVQNeuron neuron(row, col);
        som->getNeuron(neuron);

        weights = neuron.weights;
    }

    int getWinner(const MatVector<float> &sample) {

        LVQNeuron bmu;
        som->findBMU(sample, bmu);
        return bmu.c + bmu.r * som->getSomCols();
    }

    void getWinners(const MatVector<float> &sample, std::vector<int> &winner) {

        winner.clear();

        LVQNeuron bmu;
        som->findBMU(sample, bmu);
        winner.push_back(bmu.c + bmu.r * som->getSomCols());
    }
};
#endif

#ifdef _DSSOMC_H

class ClusteringDSSOMC : public ClusteringSOM<DSSOMC<DSCNeuron, DSSOMCParameters> > {
public:

    ClusteringDSSOMC(DSSOMC<DSCNeuron, DSSOMCParameters> *som) : ClusteringSOM<DSSOMC<DSCNeuron, DSSOMCParameters> >(som) {
    };

    int getMeshSize() {
        return som->getSomCols() * som->getSomRows();
    }

    void train(MatMatrix<float> &trainingData, int N) {
        som->parameters.NFeatures = trainingData.cols();
        som->parameters.tmax = N;
        som->initializeP(som->parameters);
        som->train(trainingData);
    }

    void getRelevances(int node_i, MatVector<float> &relevances) {
        int col = node_i / som->getSomRows();
        int row = ((node_i / (float) som->getSomRows()) - col) * som->getSomRows();

        DSCNeuron neuron(row, col);
        som->getNeuron(neuron);

        relevances = neuron.dsWeights;
    }

    void getWeights(int node_i, MatVector<float> &weights) {
        int col = node_i / som->getSomRows();
        int row = ((node_i / (float) som->getSomRows()) - col) * som->getSomRows();

        DSCNeuron neuron(row, col);
        som->getNeuron(neuron);

        weights = neuron.weights;
    }

    int getWinner(const MatVector<float> &sample) {

        DSCNeuron bmu;
        som->findBMU(sample, bmu);
        return bmu.c + bmu.r * som->getSomCols();
    }

    void getWinners(const MatVector<float> &sample, std::vector<int> &winner) {

        winner.clear();

        DSCNeuron bmu;
        som->findBMU(sample, bmu);
        winner.push_back(bmu.c + bmu.r * som->getSomCols());
    }
};
#endif

#ifdef _SOMAW_H

class ClusteringSOMAW : public ClusteringSOM<SOMAW<DSNeuron, SOMAWParameters> > {
public:

    ClusteringSOMAW(SOMAW<DSNeuron, SOMAWParameters> *som) : ClusteringSOM<SOMAW<DSNeuron, SOMAWParameters> >(som) {
    };

    int getMeshSize() {
        return som->getSomCols() * som->getSomRows();
    }

    void train(MatMatrix<float> &trainingData, int N) {
        som->parameters.NFeatures = trainingData.cols();
        som->parameters.tmax = N;
        som->initializeP(som->parameters);
        som->train(trainingData);
    }

    void getRelevances(int node_i, MatVector<float> &relevances) {
        int col = node_i / som->getSomRows();
        int row = ((node_i / (float) som->getSomRows()) - col) * som->getSomRows();

        DSNeuron neuron(row, col);
        som->getNeuron(neuron);

        relevances = neuron.dsWeights;
    }

    void getWeights(int node_i, MatVector<float> &weights) {
        int col = node_i / som->getSomRows();
        int row = ((node_i / (float) som->getSomRows()) - col) * som->getSomRows();

        DSNeuron neuron(row, col);
        som->getNeuron(neuron);

        weights = neuron.weights;
    }

    int getWinner(const MatVector<float> &sample) {

        DSNeuron bmu;
        som->findBMU(sample, bmu);
        return bmu.c + bmu.r * som->getSomCols();
    }

    void getWinners(const MatVector<float> &sample, std::vector<int> &winner) {

        winner.clear();

        DSNeuron bmu;
        som->findBMU(sample, bmu);
        winner.push_back(bmu.c + bmu.r * som->getSomCols());
    }
};
#endif

#ifdef _DSSOM_H

class ClusteringDSSOM : public ClusteringSOM<DSSOM<DSNeuron, DSSOMParameters> > {
public:

    ClusteringDSSOM(DSSOM<DSNeuron, DSSOMParameters> *som) : ClusteringSOM<DSSOM<DSNeuron, DSSOMParameters> >(som) {
    };

    int getMeshSize() {
        return som->getSomCols() * som->getSomRows();
    }

    void train(MatMatrix<float> &trainingData, int N) {
        som->parameters.NFeatures = trainingData.cols();
        som->parameters.tmax = N;
        som->initializeP(som->parameters);
        som->train(trainingData);
    }

    void getRelevances(int node_i, MatVector<float> &relevances) {
        int col = node_i / som->getSomRows();
        int row = ((node_i / (float) som->getSomRows()) - col) * som->getSomRows();

        DSNeuron neuron(row, col);
        som->getNeuron(neuron);
        relevances = neuron.dsWeights;
    }

    void getWeights(int node_i, MatVector<float> &weights) {
        int col = node_i / som->getSomRows();
        int row = ((node_i / (float) som->getSomRows()) - col) * som->getSomRows();

        DSNeuron neuron(row, col);
        som->getNeuron(neuron);

        weights = neuron.weights;
    }

    int getWinner(const MatVector<float> &sample) {

        DSNeuron bmu;
        som->findBMU(sample, bmu);
        return bmu.c + bmu.r * som->getSomCols();
    }

    void getWinners(const MatVector<float> &sample, std::vector<int> &winner) {

        winner.clear();

        DSNeuron bmu;
        som->findBMU(sample, bmu);
        winner.push_back(bmu.c + bmu.r * som->getSomCols());
    }
};
/*
class ClusteringDSSOM: public ClusteringSOM<DSSOM> {

public:
    ClusteringDSSOM(DSSOM *som) : ClusteringSOM<DSSOM>(som){};
    
    int getMeshSize() {
        return som->getSomCols() * som->getSomRows();
    }

    void train(MatMatrix<float> &trainingData, int N) {
        som->parameters.NFeatures = trainingData.cols();
        som->parameters.tmax = N;
        som->initializeP(som->parameters);
        som->train(trainingData);
    }
    
    void getRelevances(int node_i, MatVector<float> &relevances) {
        int col = node_i / som->getSomRows();
        int row = ((node_i / (float) som->getSomRows()) - col) * som->getSomRows();

        DSNeuron neuron(row, col);
        som->getNeuron(neuron);

        relevances = neuron.dsWeights;
    }

    void getWeights(int node_i, MatVector<float> &weights) {
        int col = node_i / som->getSomRows();
        int row = ((node_i / (float) som->getSomRows()) - col) * som->getSomRows();

        DSNeuron neuron(row, col);
        som->getNeuron(neuron);

        weights = neuron.dsWeights;
    }
    
    int getWinner(const MatVector<float> &sample) {
        
        DSNeuron bmu;
        som->findFirstBMU(sample, bmu);
        return bmu.c + bmu.r*som->getSomCols();
    }
        
    void getWinners(const MatVector<float> &sample, std::vector<int> &winner) {
    
        DSNeuron bmu;
        som->resetRelevances();
        int k=1;
        while (som->getRelevance().max()>som->parameters.epsilonRho && k<=som->parameters.numWinners)
        {
            if (k==1) {
                float activation = som->findFirstBMU(sample, bmu);
                if (activation < som->parameters.outliersThreshold) {
                    winner.push_back(-1);
                    break;
                }
            }
            else {
                som->findNextBMU(sample, bmu);
            }
            winner.push_back(bmu.c + bmu.r*som->getSomCols());
            som->updateRelevances(bmu);
            k = k + 1;
        }
    }
};
 */
#endif

#ifdef SOM_H_

class ClusteringMeshSOM : public ClusteringSOM<SOM<DSNode> > {
public:

    ClusteringMeshSOM(SOM<DSNode> *som) : ClusteringSOM<SOM<DSNode> >(som) {
    };

    int getMeshSize() {
        return som->meshNodeSet.size();
    }

    void train(MatMatrix<float> &trainingData, int N) {
        som->data = trainingData;
        som->trainning(N);
    }

    void getRelevances(int node_i, MatVector<float> &relevances) {

        SOM<DSNode>::TPNodeSet::iterator it = som->meshNodeSet.begin();
        int i = 0;
        for (; it != som->meshNodeSet.end(); it++, i++) {
            if (i == node_i) {
                relevances = (*it)->ds;
                return;
            }
        }
        return;
    }

    void getWeights(int node_i, MatVector<float> &weights) {
        SOM<DSNode>::TPNodeSet::iterator it = som->meshNodeSet.begin();
        int i = 0;
        for (; it != som->meshNodeSet.end(); it++, i++) {
            if (i == node_i) {
                weights = (*it)->w;
                return;
            }
        }
        return;
    }

    int getWinner(const MatVector<float> &sample) {
        DSNode *winner = som->getWinner(sample);
        return getNodeIndex(*winner);
    }

    void getWinners(const MatVector<float> &sample, std::vector<int> &winners) {
        DSNode *winner = som->getFirstWinner(sample);
        winners.push_back(getNodeIndex(*winner));

        winner = som->getNextWinner(winner);
        while (winner != NULL) {
            winners.push_back(getNodeIndex(*winner));
            winner = som->getNextWinner(winner);
        }
    }

    virtual bool isNoise(const MatVector<float> &sample) {
        return som->isNoise(sample);
    }

    int getNodeIndex(DSNode &node) {
        SOM<DSNode>::TPNodeSet::iterator it = som->meshNodeSet.begin();
        int i = 0;
        for (; it != som->meshNodeSet.end(); it++, i++) {
            if ((*it) == &node) {
                return i;
            }
        }
        return -1;
    }

    int getNodeId(int node_i) {
        SOM<DSNode>::TPNodeSet::iterator it = som->meshNodeSet.begin();
        int i = 0;
        for (; it != som->meshNodeSet.end(); it++, i++) {
            if (i == node_i) {
                return (*it)->getId();
            }
        }
        return -1;
    }
};

class ClusteringMeshSOMNodeW : public ClusteringSOM<SOM<NodeW> > {
public:

    ClusteringMeshSOMNodeW(SOM<NodeW> *som) : ClusteringSOM<SOM<NodeW> >(som) {
    };

    int getMeshSize() {
        return som->meshNodeSet.size();
    }

    void train(MatMatrix<float> &trainingData, int N) {
        som->data = trainingData;
        som->trainning(N);
    }

    void getRelevances(int node_i, MatVector<float> &relevances) {
        SOM<NodeW>::TPNodeSet::iterator it = som->meshNodeSet.begin();
        relevances.size((*it)->w.size());
        relevances.fill(1);
        return;
    }

    void getWeights(int node_i, MatVector<float> &weights) {
        SOM<NodeW>::TPNodeSet::iterator it = som->meshNodeSet.begin();
        int i = 0;
        for (; it != som->meshNodeSet.end(); it++, i++) {
            if (i == node_i) {
                weights = (*it)->w;
                return;
            }
        }
        return;
    }

    int getWinner(const MatVector<float> &sample) {
        NodeW *winner = som->getWinner(sample);
        return getNodeIndex(*winner);
    }

    void getWinners(const MatVector<float> &sample, std::vector<int> &winner) {
        NodeW *winner1 = 0;
        NodeW *winner2 = 0;

        som->getWinners(sample, winner1, winner2);

        winner.push_back(getNodeIndex(*winner1));
        winner.push_back(getNodeIndex(*winner2));
    }

    int getNodeIndex(Node &node) {
        SOM<NodeW>::TPNodeSet::iterator it = som->meshNodeSet.begin();
        int i = 0;
        for (; it != som->meshNodeSet.end(); it++, i++) {
            if ((*it) == &node) {
                return i;
            }
        }
        return -1;
    }
};
#endif

#endif /* CLUSTERINGSOM_H */
