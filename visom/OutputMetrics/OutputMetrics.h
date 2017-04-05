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
#include "VILARFDSSOM.h"
#include "unistd.h"
#include "MyParameters/MyParameters.h"
#include "OutputMetrics/OutputMetrics.h"
#include <string>
using namespace std;

class OutputMetrics {
public:

    void output(VILARFDSSOM *som, int experiment, MatMatrix<int> taxaTrue, MatMatrix<int> taxaFalse) {
        std::ofstream file1;
        std::string name = "output/metrics.txt";
        file1.open(name.c_str(), std::ios_base::app);

        file1 << "Params:" << endl;
        file1 << "a_t " << " = " << som->a_t << "  ";
        file1 << "dsbeta " << " = " << som->dsbeta << "  ";
        file1 << "e_b " << " = " << som->e_b << "  ";
        file1 << "e_n " << " = " << som->e_n << "  ";
        file1 << "epsilon_ds " << " = " << som->epsilon_ds << "  ";
        file1 << "lp " << " = " << som->lp << "  ";
        file1 << "minwd " << " = " << som->minwd << endl;
        file1 << "Experimento = " << experiment;

        std::ofstream file2;
        std::string name2 = "output/metrics_read.txt";
        file2.open(name2.c_str(), std::ios_base::app);

        file2 << "\n\nParams:" << endl;
        file2 << "a_t " << " = " << som->a_t << "  ";
        file2 << "dsbeta " << " = " << som->dsbeta << "  ";
        file2 << "e_b " << " = " << som->e_b << "  ";
        file2 << "e_n " << " = " << som->e_n << "  ";
        file2 << "epsilon_ds " << " = " << som->epsilon_ds << "  ";
        file2 << "lp " << " = " << som->lp << "  ";
        file2 << "minwd " << " = " << som->minwd << endl;
        file2 << "Experimento = " << experiment;

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

    void outputWithParamsFiles(VILARFDSSOM *som, int experiment, MatMatrix<int> taxaTrue, MatMatrix<int> taxaFalse, MyParameters params, int fileNumber) {
        std::ofstream file1;
        std::string name = "output/metrics.txt";
        file1.open(name.c_str(), std::ios_base::app);

        file1 << "Params:" << endl;
        file1 << "a_t " << " = " << som->a_t << "  ";
        file1 << "dsbeta " << " = " << som->dsbeta << "  ";
        file1 << "e_b " << " = " << som->e_b << "  ";
        file1 << "e_n " << " = " << som->e_n << "  ";
        file1 << "epsilon_ds " << " = " << som->epsilon_ds << "  ";
        file1 << "lp " << " = " << som->lp << "  ";
        file1 << "minwd " << " = " << som->minwd << endl;
        file1 << "Experimento = " << experiment;

        std::ofstream file2;
        std::string name2 = "output/metrics_read.txt";
        file2.open(name2.c_str(), std::ios_base::app);

        file2 << "\n\nParams:" << endl;
        file2 << "a_t " << " = " << som->a_t << "  ";
        file2 << "dsbeta " << " = " << som->dsbeta << "  ";
        file2 << "e_b " << " = " << som->e_b << "  ";
        file2 << "e_n " << " = " << som->e_n << "  ";
        file2 << "epsilon_ds " << " = " << som->epsilon_ds << "  ";
        file2 << "lp " << " = " << som->lp << "  ";
        file2 << "minwd " << " = " << som->minwd << endl;
        file2 << "Experimento = " << experiment;

        //Params files
        std::ofstream file3;
        std::string name3 = "output/" + params.a_t.name + ".txt";
        file3.open(name3.c_str(), std::ios_base::app);
        file3 << "---------------------- " << params.a_t.name << " = " << som->a_t << " ----------------------" << endl;
        file3 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;

        std::ofstream file4;
        std::string name4 = "output/" + params.dsbeta.name + ".txt";
        file4.open(name4.c_str(), std::ios_base::app);
        file4 << "---------------------- " << params.dsbeta.name << " = " << som->a_t << " ----------------------" << endl;
        file4 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;

        std::ofstream file5;
        std::string name5 = "output/" + params.e_b.name + ".txt";
        file5.open(name5.c_str(), std::ios_base::app);
        file5 << "---------------------- " << params.e_b.name << " = " << som->a_t << " ----------------------" << endl;
        file5 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;

        std::ofstream file6;
        std::string name6 = "output/" + params.e_n.name + ".txt";
        file6.open(name6.c_str(), std::ios_base::app);
        file6 << "---------------------- " << params.e_n.name << " = " << som->a_t << " ----------------------" << endl;
        file6 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;

        std::ofstream file7;
        std::string name7 = "output/" + params.epsilon_ds.name + ".txt";
        file7.open(name7.c_str(), std::ios_base::app);
        file7 << "---------------------- " << params.epsilon_ds.name << " = " << som->a_t << " ----------------------" << endl;
        file7 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;

        std::ofstream file8;
        std::string name8 = "output/" + params.lp.name + ".txt";
        file8.open(name8.c_str(), std::ios_base::app);
        file8 << "---------------------- " << params.lp.name << " = " << som->a_t << " ----------------------" << endl;
        file8 << "Arquivo = " << fileNumber << " | Experimento = " << experiment << endl;

        std::ofstream file9;
        std::string name9 = "output/" + params.minwd.name + ".txt";
        file9.open(name9.c_str(), std::ios_base::app);
        file9 << "---------------------- " << params.minwd.name << " = " << som->a_t << " ----------------------" << endl;
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
            file1 << (2 * precision * recall) / (precision + recall + 0.0000000001) << endl;

            file2 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001);
            //Params files
            file3 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
            file4 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
            file5 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
            file6 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
            file7 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
            file8 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;
            file9 << precision << "  " << recall << "  " << (2 * precision * recall) / (precision + recall + 0.00000000001) << endl;


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

