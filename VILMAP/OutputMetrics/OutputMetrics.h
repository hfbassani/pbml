/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   OutputMetrics.h
 * Author: raphael
 *
 * Created on April 3, 2017, 2:19 PM
 */

#ifndef OUTPUTMETRICS_H
#define OUTPUTMETRICS_H

#include <stdlib.h>
#include <fstream>
#include "MatMatrix.h"
#include "MatVector.h"
#include "ClusteringSOM.h"
#include "VILMAP.h"
#include "unistd.h"
#include "MyParameters/MyParameters.h"
#include "OutputMetrics/OutputMetrics.h"
#include <string>
#include <sys/stat.h> 
using namespace std;

class OutputMetrics {
public:
    std::string PATH = "output/";

    void output(VILMAP *som, int experiment, MatMatrix<int> taxaTrue, MatMatrix<int> taxaFalse, int fileNumber) {
        std::ofstream file1;
        std::string name = PATH + "metrics.txt";
        mkdir(PATH.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); 
        file1.open(name.c_str(), std::ios_base::app);

        file1 << "Params:" << endl;
        file1 << "a_t " << " = " << som->a_t << "  ";
        file1 << "dsbeta " << " = " << som->dsbeta << "  ";
        file1 << "e_b " << " = " << som->e_b << "  ";
        file1 << "e_n " << " = " << som->e_n << "  ";
        file1 << "epsilon_ds " << " = " << som->epsilon_ds << "  ";
        file1 << "lp " << " = " << som->lp << "  ";
        file1 << "minwd " << " = " << som->minwd << endl;
        file1 << "Experimento = " << experiment<< "  ";
        file1 << "Arquivo = " << fileNumber;

        std::ofstream file2;
        std::string name2 = PATH + "metrics_read.txt";
        file2.open(name2.c_str(), std::ios_base::app);

        file2 << "\n\nParams:" << endl;
        file2 << "a_t " << " = " << som->a_t << "  ";
        file2 << "dsbeta " << " = " << som->dsbeta << "  ";
        file2 << "e_b " << " = " << som->e_b << "  ";
        file2 << "e_n " << " = " << som->e_n << "  ";
        file2 << "epsilon_ds " << " = " << som->epsilon_ds << "  ";
        file2 << "lp " << " = " << som->lp << "  ";
        file2 << "minwd " << " = " << som->minwd << endl;
        file2 << "Experimento = " << experiment<< "  ";
        file2 << "Arquivo = " << fileNumber;

        for (int row = 0, d = som->d_min; row < taxaTrue.rows(); row++, d++) {

            file1 << "\nTamanho_da_entrada " << d << endl;
            file2 << "\nTamanho_da_entrada " << d << endl;

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

        }
        file1.close();
        file2.close();
    }

    void outputWithParamsFiles(VILMAP *som, int experiment, MatMatrix<int> taxaTrue, MatMatrix<int> taxaFalse, int fileNumber) {
        std::ofstream file1;
        std::string name = PATH + "metrics.txt";
        mkdir(PATH.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); 
        file1.open(name.c_str(), std::ios_base::app);

        file1 << "Params:" << endl;
        file1 << "a_t " << " = " << som->a_t << "  ";
        file1 << "dsbeta " << " = " << som->dsbeta << "  ";
        file1 << "e_b " << " = " << som->e_b << "  ";
        file1 << "e_n " << " = " << som->e_n << "  ";
        file1 << "epsilon_ds " << " = " << som->epsilon_ds << "  ";
        file1 << "lp " << " = " << som->lp << "  ";
        file1 << "minwd " << " = " << som->minwd << endl;
        file1 << "Experimento = " << experiment<< "  ";
        file1 << "Arquivo = " << fileNumber;

        std::ofstream file2;
        std::string name2 = PATH + "metrics_read.txt";
        file2.open(name2.c_str(), std::ios_base::app);

        file2 << "\n\nParams:" << endl;
        file2 << "a_t " << " = " << som->a_t << "  ";
        file2 << "dsbeta " << " = " << som->dsbeta << "  ";
        file2 << "e_b " << " = " << som->e_b << "  ";
        file2 << "e_n " << " = " << som->e_n << "  ";
        file2 << "epsilon_ds " << " = " << som->epsilon_ds << "  ";
        file2 << "lp " << " = " << som->lp << "  ";
        file2 << "minwd " << " = " << som->minwd << endl;
        file2 << "Experimento = " << experiment<< "  ";
        file2 << "Arquivo = " << fileNumber;

        //Params files
        std::ofstream file3;
        std::string name3 = PATH + "a_t" + ".txt";
        file3.open(name3.c_str(), std::ios_base::app);
        file3 << "---------------------- " << "a_t" << " = " << som->a_t << " ----------------------" << endl;
        file3 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;

        std::ofstream file4;
        std::string name4 = PATH + "dsbeta" + ".txt";
        file4.open(name4.c_str(), std::ios_base::app);
        file4 << "---------------------- " << "dsbeta" << " = " << som->a_t << " ----------------------" << endl;
        file4 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;

        std::ofstream file5;
        std::string name5 = PATH + "e_b" + ".txt";
        file5.open(name5.c_str(), std::ios_base::app);
        file5 << "---------------------- " << "e_b" << " = " << som->a_t << " ----------------------" << endl;
        file5 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;

        std::ofstream file6;
        std::string name6 = PATH + "e_n" + ".txt";
        file6.open(name6.c_str(), std::ios_base::app);
        file6 << "---------------------- " << "e_n" << " = " << som->a_t << " ----------------------" << endl;
        file6 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;

        std::ofstream file7;
        std::string name7 = PATH + "epsilon_ds" + ".txt";
        file7.open(name7.c_str(), std::ios_base::app);
        file7 << "---------------------- " << "epsilon_ds" << " = " << som->a_t << " ----------------------" << endl;
        file7 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;

        std::ofstream file8;
        std::string name8 = PATH + "lp" + ".txt";
        file8.open(name8.c_str(), std::ios_base::app);
        file8 << "---------------------- " << "lp" << " = " << som->a_t << " ----------------------" << endl;
        file8 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;

        std::ofstream file9;
        std::string name9 = PATH + "minwd" + ".txt";
        file9.open(name9.c_str(), std::ios_base::app);
        file9 << "---------------------- " << "minwd" << " = " << som->a_t << " ----------------------" << endl;
        file9 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;


        for (int row = 0, d = som->d_min; row < taxaTrue.rows(); row++, d++) {

            file1 << "\nTamanho_da_entrada " << d << endl;
            file2 << "\nTamanho_da_entrada " << d << endl;
            //Params files
            file3 << "Tamanho_da_entrada " << d << endl;
            file4 << "Tamanho_da_entrada " << d << endl;
            file5 << "Tamanho_da_entrada " << d << endl;
            file6 << "Tamanho_da_entrada " << d << endl;
            file7 << "Tamanho_da_entrada " << d << endl;
            file8 << "Tamanho_da_entrada " << d << endl;
            file9 << "Tamanho_da_entrada " << d << endl;


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
            float f = (2 * precision * recall) / (precision + recall + 0.0000000001);
            file1 << f << endl;

            file2 << precision << "  " << recall << "  " << f;
            //Params files
            file3 << precision << "  " << recall << "  " << f << endl;
            file4 << precision << "  " << recall << "  " << f << endl;
            file5 << precision << "  " << recall << "  " << f << endl;
            file6 << precision << "  " << recall << "  " << f << endl;
            file7 << precision << "  " << recall << "  " << f << endl;
            file8 << precision << "  " << recall << "  " << f << endl;
            file9 << precision << "  " << recall << "  " << f << endl;


        }
        file1.close();
        file2.close();
        file3.close();
        file4.close();
        file5.close();
        file6.close();
        file7.close();
        file8.close();
        file9.close();
    }


};

#endif /* OUTPUTMETRICS_H */
