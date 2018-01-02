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
    LHSParameter pushRate;
    LHSParameter supervisionRate;
    LHSParameter seed;
    
    // LVQ
    LHSParameter nnodes;
    LHSParameter at_p;
    LHSParameter at_n;
    LHSParameter at_w;
    LHSParameter lvq_tau;
    LHSParameter lvq_epochs;
    LHSParameter lvq_seed;

    MyParameters(bool real) {
        comments = "Test float Parameters";
        section = "LARFDSSOM Parameters";

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
        addParameterD(pushRate, "pushRate rate");
        addParameterD(supervisionRate, "supervision rate");
        addParameterD(seed, "seed");
        
        addParameterD(nnodes, "Number of nodes created in map");
        addParameterD(at_p, "Learning rate positive");
        addParameterD(at_n, "Learning rate negative");
        addParameterD(at_w, "Learning rate relevances");
        addParameterD(lvq_tau, "decay rate");
        addParameterD(lvq_epochs, "LVQ Epochs");
        addParameterD(lvq_seed, "LVQ Seed");

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
        pushRate.setRange(0, 0.2) = 0;
        supervisionRate.setRange(0, 1) = 0;
        seed.setRange(0, 100) = 0;
        
        nnodes.setRange(10, 30) = 10;
        at_p.setRange(0.4, 0.5) = 0.4;
        at_n.setRange(0.01, 0.05) = 0.01;
        at_w.setRange(0.15, 0.2) = 0.15;
        lvq_tau.setRange(0.000001, 0.00002) = 0.000001;
        lvq_epochs.setRange(5000, 10000) = 5000;
        lvq_seed.setRange(0, 10000) = 0;

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
        addParameterToLHS(pushRate);
        addParameterToLHS(supervisionRate);
        addParameterToLHS(seed);
        
        addParameterToLHS(nnodes);
        addParameterToLHS(at_p);
        addParameterToLHS(at_n);
        addParameterToLHS(at_w);
        addParameterToLHS(lvq_tau);
        addParameterToLHS(lvq_epochs);
        addParameterToLHS(lvq_seed);
    }
};

#endif /* MYPARAMETERS_H */
