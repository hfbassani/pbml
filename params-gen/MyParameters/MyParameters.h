/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   MyParameters.h
 * Author: raphael
 *
 * Created on September 28, 2016, 12:08 PM
 */

#ifndef MYPARAMETERS_H
#define MYPARAMETERS_H

#include "LHSParameters.h"

class MyParameters : public LHSParameters {
public:

    Parameter<int> N;
    LHSParameter a_t;
    LHSParameter lp;
    LHSParameter dsbeta;
    LHSParameter age_wins;
    LHSParameter e_b;
    LHSParameter e_n;
    LHSParameter epsilon_ds;
    LHSParameter minwd;
    LHSParameter epochs;
    LHSParameter gamma;
    LHSParameter h_threshold;
    LHSParameter tau;

    MyParameters(bool real) {
        comments = "Test float Parameters";
        section = "VILARFDSSOM Parameters";

        //make parameters persistent:
        addParameterD(N, "Number of samples");
        addParameterD(a_t, "Activation");
        addParameterD(lp, "Cluster Percentage");
        addParameterD(dsbeta, "Relevance rate");
        addParameterD(age_wins, "Max competitions");
        addParameterD(e_b, "learning rate");
        addParameterD(e_n, "Cluster Percentage");
        addParameterD(epsilon_ds, "Relevance rate");
        addParameterD(minwd, "Relevance rate");
        addParameterD(epochs, "Epochs");
        addParameterD(gamma, "decay rate for function h");
        addParameterD(h_threshold, "threshold for function h");
        addParameterD(tau, "decay rate");

        //Set default ranges and values
        N = 150;
        if (real) {
            a_t.setRange(0.70, 0.999) = 0.70;
            lp.setRange(0.01, 0.1) = 0.01;
            dsbeta.setRange(0.001, 0.5) = 0.001;
            age_wins.setRange(1, 100) = 1;
            e_b.setRange(0.0001, 0.01) = 0.0001;
            e_n.setRange(0.002, 1.0) = 0.002;
            epsilon_ds.setRange(0.01, 0.1) = 0.01;
            minwd.setRange(0.001, 0.5) = 0.001;
        } else {
            a_t.setRange(0.90, 0.999) = 0.9;
            lp.setRange(0.0001, 0.1) = 0.0001;
            dsbeta.setRange(0.0001, 0.5) = 0.0001;
            age_wins.setRange(1, 100) = 1;
            e_b.setRange(0.0001, 0.01) = 0.0001;
            e_n.setRange(0.001, 0.1) = 0.001;
            epsilon_ds.setRange(0.01, 0.05) = 0.01;
            minwd.setRange(0, 0.5) = 0;
        }
        
        epochs.setRange(1, 100) = 100;
        gamma.setRange(0.14, 4.0) = 0.14;
        h_threshold.setRange(0.001, 0.8) = 0.001;
        tau.setRange(0.00001, 0.01) = 0.00001;

        //Add parameters to latin hypercube sampling:
        addParameterToLHS(a_t);
        addParameterToLHS(lp);
        addParameterToLHS(dsbeta);
        addParameterToLHS(age_wins);
        addParameterToLHS(e_b);
        addParameterToLHS(e_n);
        addParameterToLHS(epsilon_ds);
        addParameterToLHS(minwd);
        addParameterToLHS(epochs);
        addParameterToLHS(gamma);
        addParameterToLHS(h_threshold);
        addParameterToLHS(tau);
    }

};

#endif /* MYPARAMETERS_H */
