/* 
 * File:   SSCDataFile.cpp
 * Author: hans
 * 
 * Created on 4 de Outubro de 2011, 16:35
 */

#include "SSCDataFile.h"
#include "Cluster.h"
#include <cstdlib>
#include <iostream>
#include "Cluster.h"
#include <fstream>
#include <algorithm>
#include <vector>

using namespace std;

void SSCDataFile::createSSCDataFile(const SSCDGParameters& parameters) {
    Dimensions dimensions(parameters.d);
    dimensions.fill(1);
    vector<Cluster> clusters(parameters.k, Cluster(&dimensions));
    double data;

    ofstream dataout;
    dataout.open(parameters.outfilename.value.c_str());

    srand(time(0));

    //*/generate random di indexes for each cluster
    for (int dim=0; dim<parameters.di; dim++)
         for (int i = 0; i < parameters.k; i++) {
            int index = dimensions.getRandNotChosen();
            if (index>=0)
                clusters[i].addDimension(index);
            else
                clusters[i].addDimension(dimensions.getRandom());
        }
    /**/

    //generate dimension random gaussian parameters for each cluster
    for (int i = 0; i < parameters.k; i++) {
        clusters[i].generateRandomMSD(parameters.minj, parameters.sdmin, parameters.sdmax);
    }

    //Generate random clusters sizes
    int ntotal=0;
    for (int i = 0; i < parameters.k; i++) {
        clusters[i].n = randomUniform(parameters.Ni0, parameters.Nimax);
        ntotal+=clusters[i].n;
    }

    //reduce or increase the size of each cluster to achieve a total of N
    int n;
    for (n=ntotal; n!=parameters.N;) {
        for (int i = 0; i < parameters.k; i++) {
            if (n<parameters.N) {
                clusters[i].n++;
                n++;
            } else if (n>parameters.N) {
                clusters[i].n--;
                n--;
            }
            else break;
        }
    }

    //randomize data error lines
    vector<bool> dataError(parameters.N, false);
    vector<bool>::iterator it;
    for (int i=0; i<parameters.N*parameters.e; i++)
        dataError[i]=true;
    random_shuffle(dataError.begin(), dataError.end());

    /*Printe metadata */
    dataout << "#Parameters:" << endl;
    dataout << "#N:\t" << parameters.N << endl;
    dataout << "#d:\t" << parameters.d << endl;
    dataout << "#k:\t" << parameters.k << endl;
    dataout << "#Ni0:\t" << parameters.Ni0 << endl;
    dataout << "#Nimax:\t" << parameters.Nimax << endl;
    dataout << "#di:\t" << parameters.di << endl;
    dataout << "#minj:\t" << parameters.minj<< endl;
    dataout << "#maxj:\t" << parameters.maxj << endl;
    dataout << "#sdmin:\t" << parameters.sdmin << endl;
    dataout << "#sdmax:\t" << parameters.sdmax << endl;
    dataout << "#e:\t" << parameters.e << endl;

    dataout << "#Outlier lines:";
    for (int i=0; i<parameters.N; i++)
        if (dataError[i]) dataout << "\t" << i;
    dataout << endl;

    dataout << "#Dimension ranges:";
    for (int i=0; i<parameters.d; i++)
        dataout << "\t" << dimensions.range(i);
    dataout << endl;

    //generate data rows
    dataout << "#DataRows:\t";
    for (int dim=0; dim<parameters.d; dim++) {
        dataout << "d" << dim;
        dataout << "\t";
    }
    dataout << "cls" << endl;

    //Generate data samples
    int r=0;
    int numErrors = 0;
    for (int i = 0; i < parameters.k; i++) { // for each cluster
        for (int j = 0; j<clusters[i].n; j++) {// for each data row

            if (dataError[r]) {
                for (int dim=0; dim<parameters.d; dim++) {// for each dimension
                   data = clusters[i].generateUniformRandomData(dim, parameters.minj);
                   dataout << data << '\t';
                }
            } else {
                for (int dim=0; dim<parameters.d; dim++) { // for each dimension
                   data = clusters[i].generateGaussianRandomData(dim, parameters.minj);
                   dataout << data << '\t';
                }
            }

            if (dataError[r]) {
                numErrors++;
                dataout << parameters.k << endl;
            }
            else
                dataout << i << endl;

            r++;
        }
    }

    dataout.close();

    //print dimension ranges
    dbgOut(1) << "Dimension ranges:";
    for (int dim=0; dim<parameters.d; dim++)
        dbgOut(1) << "\t" << dimensions.range(dim);
    dbgOut(1) << endl << endl;

    //print cluster information
    for (int i = 0; i < parameters.k; i++) {
        dbgOut(1) << "cluster " << i << ":" << endl;
        dbgOut(1) << "Num sample rows:" << clusters[i].n << endl;
        std::vector<ClusterDimension> dimensions = clusters[i].getDimensions();
        dbgOut(1) << "dims(" << dimensions.size() << "):";
        for (std::vector<ClusterDimension>::iterator it = dimensions.begin(); it!= dimensions.end(); it++) {
            dbgOut(1) << it->index << "[" << it->m << "," << it->sd << "] ";
        }
        dbgOut(1) << endl << endl;
    }

    dbgOut(1) << "Cluster rows: " << n << endl;
    dbgOut(1) << "Data error rate: " << numErrors << "/" << n << " (" << 100*numErrors/(float)n << "%)" << endl << endl;

    dbgOut(1) << "Clusters dimension sumary:" << endl;
    for (int dim=0; dim<parameters.d; dim++)
    {
        dbgOut(1) << "d" << dim << "[";
        bool first = true;
        for (int i = 0; i < parameters.k; i++)
            if (clusters[i].dimIndexExists(dim)) {
                if (!first) dbgOut(1) << " ";
                first = false;
                dbgOut(1) << i;
            }
        dbgOut(1) << "] ";
    }

    dbgOut(1) << endl;
}