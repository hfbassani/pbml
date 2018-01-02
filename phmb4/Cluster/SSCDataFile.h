/* 
 * File:   SSCDataFile.h
 * Author: hans
 *
 * Created on 4 de Outubro de 2011, 16:35
 */

#ifndef SSCDATAFILE_H
#define	SSCDATAFILE_H

#include "Parameters.h"

class SSCDGParameters: public Parameters {
public:
    Parameter<int> N;           //Dataset size
    Parameter<int> d;           //Dataset dimensionality
    Parameter<int> k;           //Number of clusters
    Parameter<int> Ni0;         //Minimun cluster size
    Parameter<int> Nimax;       //Máximun cluster size
    Parameter<int> di;          //Average cluster dimensionality
    Parameter<double> minj;     //Domain of dimensions (min)
    Parameter<double> maxj;     //Domain of dimensions (max)
    Parameter<double> sdmin;    //Local S.D. of relevant dimensions (min)
    Parameter<double> sdmax;    //Local S.D. of relevant dimensions (max)
    Parameter<double> e;        //Artificial data error rate
    Parameter<std::string> outfilename; //Output file name

    SSCDGParameters() {

        //Comentários do início do arquivo
        comments = "SSCDGParameters Parameters";

        //Definição do nome da sessão do arquivo na qual estes parâmetros serão salvos
        section = "SSCDGParameters";

        //Parâmetros persistentes
        addParameterD(N,"Dataset size");
        addParameterD(d,"Dataset dimensionality");
        addParameterD(k,"Number of clusters");
        addParameterD(Ni0,"Minimun cluster size");
        addParameterD(Nimax,"Máximun cluster size");
        addParameterD(di,"Average cluster dimensionality");
        addParameterD(minj,"Domain of dimensions (min)");
        addParameterD(maxj,"Domain of dimensions (max)");
        addParameterD(sdmin,"Local S.D. of relevant dimensions (min)");
        addParameterD(sdmax,"Local S.D. of relevant dimensions (max)");
        addParameterD(e  ,"Artificial data error rate");
        addParameterD(outfilename,"Output file name");

        //Default values
        N = 500;           //Dataset size
        d = 20;            //Dataset dimensionality
        k = 5;             //Number of clusters
        Ni0 = (0.15*N);    //Minimun cluster size
        Nimax = (0.25*N);  //Máximun cluster size
        di = 4;            //Average cluster dimensionality
        minj = 0;          //Domain of dimensions (min)
        maxj = 1;          //Domain of dimensions (max)
        sdmin = 0.03;      //Local S.D. of relevant dimensions (min)
        sdmax = 0.05;      //Local S.D. of relevant dimensions (max)
        e   = 0.05;        //Artificial data error rate
        outfilename = "out.csv"; //Output file name
    }
};

class SSCDataFile {
public:
    void createSSCDataFile(const SSCDGParameters& parameters);
};

#endif	/* SSCDATAFILE_H */

