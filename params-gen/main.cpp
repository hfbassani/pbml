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

void createParametersFileOriginalLARFDSSOM(MyParameters * params, string fileName, int qtdParameters);
void createParametersFileHybrid(MyParameters * params, string fileName, int qtdParameters);
void createParametersFileExperiments(MyParameters * params, string fileName, int qtdParameters);
void createParametersMLP_SVM(MyParameters * params, string fileName, int qtdParameters);
void createLVQParameters(MyParameters * params, string fileName, int qtdParameters);

std::vector<float> loadParametersFile(int number);

int main(int argc, char** argv) {

    string filename = "";
    int qtd_files = 1;
    int qtd_parameter = 0;
    
    bool originalVersion = false;
    bool simulatedData = false;
    bool hybridVersion = false;

    bool SVMMLP = false;
    bool lvq = false;
    
    int c;
    while ((c = getopt(argc, argv, "f:n:r:sohdl")) != -1) {
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
            case 's':
                simulatedData = true;
                break;
            case 'o':
                originalVersion = true;
                break;
            case 'h':
                hybridVersion = true;
                break;
            case 'd':
                SVMMLP = true;
            case 'l':
                lvq = true;
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
    
    srand(time(NULL));
    
    MyParameters params(!simulatedData); //Teste de Parâmetros
    
    for (int i = 0 ; i < qtd_files ; ++i) {
        if (originalVersion) {
            createParametersFileOriginalLARFDSSOM(&params, filename + "_" + std::to_string(i), qtd_parameter);   
        } else if(hybridVersion){
            createParametersFileHybrid(&params, filename + "_" + std::to_string(i), qtd_parameter);
        } else if(SVMMLP){
            createParametersMLP_SVM(&params, filename + "_" + std::to_string(i), qtd_parameter);
        } else if(lvq){
            createLVQParameters(&params, filename + "_" + std::to_string(i), qtd_parameter);
        } else {
            createParametersFileExperiments(&params, filename + "_" + std::to_string(i), qtd_parameter);
        }
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

void createParametersFileOriginalLARFDSSOM(MyParameters * params, string fileName, int qtdParameters) {
    std::ofstream file;
    file.open(fileName.c_str());
    
    cout << "createParametersFileOriginalLARFDSSOM" << endl;
    
    for (params->initLHS(qtdParameters) ; !params->finished(); params->setNextValues()) {
        file << params->a_t.value << "\n";
        file << params->lp << "\n";
        file << params->dsbeta << "\n";
        file << round(params->age_wins) << "\n";
        file << params->e_b << "\n";
        file << params->e_n << "\n";
        file << params->epsilon_ds << "\n";
        file << params->minwd << "\n";
        file << round(params->epochs) << "\n";
        file << params->seed << "\n";
    }
    
    file.close();
}

void createParametersFileHybrid(MyParameters * params, string fileName, int qtdParameters) {
    std::ofstream file;
    file.open(fileName.c_str());
    
    cout << "createParametersFileHybrid" << endl;
    
    for (params->initLHS(qtdParameters) ; !params->finished(); params->setNextValues()) {
        file << params->a_t.value << "\n";
        file << params->lp << "\n";
        file << params->dsbeta << "\n";
        file << round(params->age_wins) << "\n";
        file << params->e_b << "\n";
        file << params->e_n << "\n";
        file << params->epsilon_ds << "\n";
        file << params->minwd << "\n";
        file << round(params->epochs) << "\n";
        file << params->pushRate << "\n";
        file << params->supervisionRate << "\n";
        file << params->seed << "\n";
    }
    
    file.close();
}

void createLVQParameters(MyParameters * params, string fileName, int qtdParameters) {
    std::ofstream file;
    file.open(fileName.c_str());

    cout << "createLVQParameters" << endl;
    
    for (params->initLHS(qtdParameters) ; !params->finished(); params->setNextValues()) {
        file << round(params->nnodes) << "\n";
        file << params->at_p << "\n";
        file << params->at_n << "\n";
        file << params->at_w << "\n";
        file << params->lvq_tau << "\n";
        file << round(params->lvq_epochs) << "\n";
        file << round(params->lvq_seed) << "\n";
    }
    
    file.close();
}

void createParametersFileExperiments(MyParameters * params, string fileName, int qtdParameters) {
    std::ofstream file;
    file.open(fileName.c_str());

    cout << "createParametersFileExperiments" << endl;
    
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
        file << params->seed << "\n";
    }
    
    file.close();
}

void createParametersMLP_SVM(MyParameters * params, string fileName, int qtdParameters) {
    std::ofstream file;
    file.open(fileName.c_str());

    cout << "createParametersMLP_SVM" << endl;
    
    for (params->initLHS(qtdParameters) ; !params->finished(); params->setNextValues()) {
        file << params->c << "\n";
        file << round(params->kernel) << "\n";
        file << round(params->degree) << "\n";
        
        
        file << round(params->neurons) << "\n";;
        file << round(params->hidden_layers) << "\n";
        file << params->lr << "\n";
        file << params->momentum << "\n";
        file << round(params->mlp_epochs) << "\n";
        file << round(params->activation) << "\n";
        file << round(params->lr_decay) << "\n";
        file << round(params->solver) << "\n";
    }
    
    file.close();
}