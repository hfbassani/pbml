/* 
 * File:   ART2Context.h
 * Author: hans
 *
 * Created on 27 de Outubro de 2009, 08:39
 */

#ifndef _ART2CONTEXT_H
#define	_ART2CONTEXT_H

#include "Parameters.h"
#include "MatMatrix.h"
#include "MatVector.h"

#include <string>
#include <math.h>


class ART2ContextParameters: public Parameters {

public:
    Parameter<int> n;
    Parameter<float> a;
    Parameter<float> b;
    Parameter<float> c;
    Parameter<float> d;
    Parameter<float> e;
    Parameter<float> theta;
    Parameter<float> alpha;
    Parameter<float> rho;
    Parameter<int> nbrEpochs;
    Parameter<int> nbrIterations;
    Parameter<float> back;
    Parameter<float> contextWeight;
    Parameter<float> d_context;
    Parameter<float> alpha_context;
    Parameter<float> denorm_weight;

    ART2ContextParameters() {

        comments = "Parametros do ART2 com Context";
        section = "ART2ContextParameters";

        addParameterD(n, "Number of input unities at F1 layer");
        addParameterD(a, "Fixed weight at F1");
        addParameterD(b, "Fixed weight at F1");
        addParameterD(c, "Fixed weights used by the reset condition in [0,1] interval");
        addParameterD(d, "Activation of the winner unity in F2");
        addParameterD(e, "Parameter introduced to avoid division by zero when the norm of a vector is zero");
        addParameterD(theta, "Parameter of noise suppression");
        addParameterD(alpha, "Learning rate");
        addParameterD(rho, "Surveillance parameter");
        addParameterD(nbrEpochs, "Low values for fast learning rate");
        addParameterD(nbrIterations, "High values for fast learning rate");
        addParameterD(back, "Context backpropagation rate");
        addParameterD(contextWeight, "Influence rate of the contextual information over the reset mechanism");
        addParameterD(d_context, "Equivalent to 'd' for the context unities");
        addParameterD(alpha_context, "Context learning rate");
        addParameterD(denorm_weight, "Denormalization weight. Denorm only if > 0");

        //Default values
        n = 183;
        a = 10;
        b = 10;
        c = 0.1;
        d = 0.9;
        e = 0.0001;
        theta = 1.0/sqrt((float)n);
        alpha = 0.001;
        rho = 1;
        nbrEpochs = 1;
        nbrIterations = 1;
        back = 0.9;
        contextWeight = 0.0;
        d_context = 0.9;
        alpha_context = 0.8;
        denorm_weight = 0.2;
    }

};


class ART2Context {

private:

public:
    ART2Context();
    virtual ~ART2Context();

    ////// Model parameters //////
    ART2ContextParameters parameters;

    ////// Model variables //////

    //Vectors
    MatVector<float> P, Q, R, S, U, V, W, X, Y, UC, PC, Context, Prototype;

    //Node in F2 with higher activation
    uint J;

    // reset
    bool reset;

    //Matrices
    MatMatrix<float> T, B;

    ////// Model functions ////// 
    void init(int n);
    void init(MatVector<float> &input);
    void initialiaze(CFGFile &parametersFile);
    void saveParameters(CFGFile &parametersFile);
    float f(float x);
    void train(MatMatrix<float> &Ss);
    void train(MatVector<float> &S);
    void recognize(MatMatrix<float> &Ss, MatVector<int> &ry);
    int recognize(MatVector<float> &S);
    void printClusters(MatVector<int> &ry, const std::string &wordsTestFileName);
    float cos(MatVector<float> &x, MatVector<float> &y);
    const MatVector<float> &getCurrentContext(bool denorm = true);
    const MatVector<float> &getWinnerPrototype(bool denorm = true);
    const MatVector<float> &getOutput(bool denorm = true);

private:

};

#endif	/* _ART2CONTEXT_H */

