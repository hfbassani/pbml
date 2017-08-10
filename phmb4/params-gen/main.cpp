/*
 * File:   main.cpp
 * Author: hans
 *
 * Created on 11 de Outubro de 2010, 07:25
 */
#define NO_INVALID_DIMENSION_SIZE
#define PRINT_CLUSTER

#include <stdlib.h>
#include <fstream>
#include <unistd.h>
#include <math.h>
#include "MyParameters/MyParameters.h"
#include <string>

using namespace std;

void createParametersFile(MyParameters * params, string fileName, int qtdParameters);

std::vector<float> loadParametersFile(int number);

int main(int argc, char** argv) {

    MyParameters params; //Teste de Parâmetros
    string filename = "";
    int qtd_files = 1;
    int qtd_parameter = 0;

    int c;
    while ((c = getopt(argc, argv, "f:n:r:")) != -1) {
        switch (c) {
            case 'f':
                filename.assign(optarg);
                break;
            case 'n':
                qtd_files = atoi(optarg);
                break;
            case 'r':
                qtd_parameter = atoi(optarg);
                break;
        }
    }
    
    cout << "filename: "  << filename << "\n";
    cout << "number of files: " << qtd_files << "\n";
    cout << "number of sets: " << qtd_parameter << "\n";
        

    if (filename == "" || qtd_parameter == 0) {
        cout << "option -f [filename] [-n number of files]-r [number of sets] is required" << endl;
        return -1;
    }

    for (int i = 0 ; i < qtd_files ; ++i) {
        createParametersFile(&params, filename + "_" + std::to_string(i), qtd_parameter);   
    } 

    cout << "Done." << endl;
    
    return 0;
}

std::vector<float> loadParametersFile(string fileName) {
    std::string text;
    std::vector<float> params;
    std::ifstream file(fileName.c_str());
    if (!file.is_open()) {
        cout << "Error openning file: " << fileName << endl;
    } else {
        while (!file.eof()) {
            getline(file, text);
            params.push_back(std::atof(text.c_str()));
        }
    }
    return params;
}

void createParametersFile(MyParameters * params, string fileName, int qtdParameters) {
    std::ofstream file;
    file.open(fileName.c_str());

    for (params->initLHS(qtdParameters) ; !params->finished(); params->setNextValues()) {
        file << params->a_t.value << "\n";
        file << params->lp << "\n";
        file << params->dsbeta << "\n";
        file << round(params->age_wins) << "\n";
        file << params->e_b << "\n";
        file << params->epsilon_ds << "\n";
        file << params->minwd << "\n";
        file << round(params->epochs) << "\n";
        file << params->gamma << "\n";
        file << params->h_threshold << "\n";
        file << params->tau << "\n";


    }
    
    file.close();
}