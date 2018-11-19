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

void createLARFDSSOMParameters(MyParameters * params, string fileName, int qtdParameters);
void createSSSOMParameters(MyParameters * params, string fileName, int qtdParameters);
void createParametersFileExperiments(MyParameters * params, string fileName, int qtdParameters);
void createMLPParameters(MyParameters * params, string fileName, int qtdParameters);
void createSVMParameters(MyParameters * params, string fileName, int qtdParameters);
void createGLVQParameters(MyParameters * params, string fileName, int qtdParameters);
void createLabelSpreadingParameters(MyParameters * params, string fileName, int qtdParameters);
void createLabelPropagationParameters(MyParameters * params, string fileName, int qtdParameters);
void createWIPParameters(MyParameters * params, string fileName, int qtdParameters);

std::vector<float> loadParametersFile(int number);

int main(int argc, char** argv) {

    string filename = "";
    int qtd_files = 1;
    int qtd_parameter = 0;
    
    bool larfdssom = false;
    bool simulatedData = false;
    bool sssom = false;
    bool wip = false;

    bool SVM = false;
    bool MLP = false;
    bool GRLVQ = false;
    bool PROP = false;
    bool SPR = false;
    
    int c;
    while ((c = getopt(argc, argv, "f:n:r:sLSWMVGPR")) != -1) {
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
            case 'L':
                larfdssom = true;
                break;
            case 'S':
                sssom = true;
                break;
            case 'W':
                wip = true;
                break;
            case 'M':
                MLP = true;
            case 'V':
                SVM = true;
            case 'G':
                GRLVQ = true;
                break;
            case 'P':
                PROP = true;
                break;
            case 'R':
                SPR = true;
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
        if (larfdssom) {
            createLARFDSSOMParameters(&params, filename + "_" + std::to_string(i), qtd_parameter);   
        } else if(sssom){
            createSSSOMParameters(&params, filename + "_" + std::to_string(i), qtd_parameter);
        } else if(MLP){
            createMLPParameters(&params, filename + "_" + std::to_string(i), qtd_parameter);
        } else if(SVM){
            createSVMParameters(&params, filename + "_" + std::to_string(i), qtd_parameter);
        } else if(GRLVQ){
            createGLVQParameters(&params, filename + "_" + std::to_string(i), qtd_parameter);
        } else if(PROP){
            createLabelPropagationParameters(&params, filename + "_" + std::to_string(i), qtd_parameter);
        } else if(SPR){
            createLabelSpreadingParameters(&params, filename + "_" + std::to_string(i), qtd_parameter);
        } else if(wip) {
            createWIPParameters(&params, filename + "_" + std::to_string(i), qtd_parameter);
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

void createLARFDSSOMParameters(MyParameters * params, string fileName, int qtdParameters) {
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
        file << round(params->seed) << "\n";
    }
    
    file.close();
}

void createSSSOMParameters(MyParameters * params, string fileName, int qtdParameters) {
    std::ofstream file;
    file.open(fileName.c_str());
    
    cout << "createParametersFileSSSOM" << endl;
    
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
        file << round(params->seed) << "\n";
    }
    
    file.close();
}

void createWIPParameters(MyParameters * params, string fileName, int qtdParameters) {
    std::ofstream file;
    file.open(fileName.c_str());
    
    cout << "createParametersFileWIP" << endl;
    
    for (params->initLHS(qtdParameters) ; !params->finished(); params->setNextValues()) {
        file << params->lp << "\n";
        file << params->dsbeta << "\n";
        file << round(params->age_wins) << "\n";
        file << params->e_b << "\n";
        file << params->e_n << "\n";
        file << params->epsilon_ds << "\n";
        file << params->minwd << "\n";
        file << round(params->epochs) << "\n";
        file << params->e_var << "\n";
        file << round(params->seed) << "\n";
    }
    
    file.close();
}

void createGLVQParameters(MyParameters * params, string fileName, int qtdParameters) {
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
        file << round(params->seed) << "\n";
    }
    
    file.close();
}

void createMLPParameters(MyParameters * params, string fileName, int qtdParameters) {
    std::ofstream file;
    file.open(fileName.c_str());

    cout << "createParametersMLP" << endl;
    
    for (params->initLHS(qtdParameters) ; !params->finished(); params->setNextValues()) {
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

void createSVMParameters(MyParameters * params, string fileName, int qtdParameters) {
    std::ofstream file;
    file.open(fileName.c_str());

    cout << "createParametersSVM" << endl;
    
    for (params->initLHS(qtdParameters) ; !params->finished(); params->setNextValues()) {
        file << params->c << "\n";
        file << round(params->kernel) << "\n";
        file << round(params->degree) << "\n";
        file << params->svm_gamma << "\n";
        file << params->coef0 << "\n";
    }
    
    file.close();
}

void createLabelSpreadingParameters(MyParameters * params, string fileName, int qtdParameters) {
    std::ofstream file;
    file.open(fileName.c_str());

    cout << "createParametersSpreading" << endl;
    
    for (params->initLHS(qtdParameters) ; !params->finished(); params->setNextValues()) {
        file << round(params->kernel_spreading) << "\n";
        file << params->gamma_spreading << "\n";
        file << round(params->neighbors_spreading) << "\n";
        file << params->alpha_spreading << "\n";
        file << round(params->epochs_spreading) << "\n";
    }
    
    file.close();
}

void createLabelPropagationParameters(MyParameters * params, string fileName, int qtdParameters) {
    std::ofstream file;
    file.open(fileName.c_str());

    cout << "createParametersPropagation" << endl;
    
    for (params->initLHS(qtdParameters) ; !params->finished(); params->setNextValues()) {
        file << round(params->kernel_propagation) << "\n";
        file << params->gamma_propagation << "\n";
        file << round(params->neighbors_propagation) << "\n";
        file << round(params->epochs_propagation) << "\n";
    }
    
    file.close();
}