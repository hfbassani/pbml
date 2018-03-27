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
    LHSParameter e_b_sup;
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
    
    //SVM
    LHSParameter c;
    LHSParameter kernel;
    LHSParameter degree;
    
    //MLP
    LHSParameter neurons;
    LHSParameter hidden_layers;
    LHSParameter lr;
    LHSParameter momentum;
    LHSParameter mlp_epochs;
    LHSParameter activation;
    LHSParameter lr_decay;
    LHSParameter solver;
    
    // LVQ
    LHSParameter nnodes;
    LHSParameter at_p;
    LHSParameter at_n;
    LHSParameter at_w;
    LHSParameter lvq_tau;
    LHSParameter lvq_epochs;
    LHSParameter lvq_seed;
    
    //Label Spreading
    LHSParameter kernel_spreading;
    LHSParameter gamma_spreading;
    LHSParameter neighbors_spreading;
    LHSParameter alpha_spreading;
    LHSParameter epochs_spreading;
    
    //Label Prop
    LHSParameter kernel_propagation;
    LHSParameter gamma_propagation;
    LHSParameter neighbors_propagation;
    LHSParameter epochs_propagation;

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
        addParameterD(e_b_sup, "learning rate");
        addParameterD(e_n, "Cluster Percentage");
        addParameterD(epsilon_ds, "Relevance rate");
        addParameterD(minwd, "Relevance rate");
        addParameterD(epochs, "Epochs");
        addParameterD(gamma, "decay rate for function h");
        addParameterD(h_threshold, "threshold for function h");
        addParameterD(tau, "decay rate");
        addParameterD(pushRate, "pushRate rate");
        addParameterD(seed, "seed");
        
        addParameterD(c, "seed");
        addParameterD(kernel, "seed");
        addParameterD(degree, "seed");
        
        addParameterD(neurons, "seed");
        addParameterD(hidden_layers, "seed");
        addParameterD(lr, "seed");
        addParameterD(momentum, "seed");
        addParameterD(mlp_epochs, "seed");
        addParameterD(activation, "seed");
        addParameterD(lr_decay, "seed");
        addParameterD(solver, "seed");

        addParameterD(nnodes, "Number of nodes created in map");
        addParameterD(at_p, "Learning rate positive");
        addParameterD(at_n, "Learning rate negative");
        addParameterD(at_w, "Learning rate relevances");
        addParameterD(lvq_tau, "decay rate");
        addParameterD(lvq_epochs, "LVQ Epochs");
        addParameterD(lvq_seed, "LVQ Seed");

        //Label Spreading
        addParameterD(kernel_spreading, "kernel_spreading");
        addParameterD(gamma_spreading, "gamma_spreading");
        addParameterD(neighbors_spreading, "neighbors_spreading");
        addParameterD(alpha_spreading, "alpha_spreading");
        addParameterD(epochs_spreading, "epochs_spreading");
    
        //Label Prop
        addParameterD(kernel_propagation, "kernel_propagation");
        addParameterD(gamma_propagation, "gamma_propagation");
        addParameterD(neighbors_propagation, "neighbors_propagation");
        addParameterD(epochs_propagation, "epochs_propagation");
    
        //Set default ranges and values
        N = 500;
        if (real) {
            a_t.setRange(0.70, 0.999) = 0.70;
            lp.setRange(0.01, 0.10) = 0.01;
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
        
        epochs.setRange(1, 100) = 1;
        gamma.setRange(0.14, 4.0) = 0.14;
        h_threshold.setRange(0.001, 0.8) = 0.001;
        tau.setRange(0.00001, 0.00004) = 0.00001;
        e_b_sup.setRange(0.01, 0.7) = 0.01;
        pushRate.setRange(0.01, 1.0) = 0.01;
        seed.setRange(1, 10000) = 1;
        
        c.setRange(0.1, 10) = 0.1;
        kernel.setRange(1, 4) = 1;
        degree.setRange(3, 5) = 3;
        
        neurons.setRange(1, 100) = 1;
        hidden_layers.setRange(1, 3) = 1;
        lr.setRange(0.001, 0.1) = 0.001;
        momentum.setRange(0.85, 0.95) = 0.85;
        mlp_epochs.setRange(100, 200) = 100;
        activation.setRange(1, 3) = 1;
        lr_decay.setRange(1, 3) = 1;
        solver.setRange(1, 3) = 1;

        nnodes.setRange(10, 30) = 10;
        at_p.setRange(0.4, 0.5) = 0.4;
        at_n.setRange(0.01, 0.05) = 0.01;
        at_w.setRange(0.15, 0.2) = 0.15;
        lvq_tau.setRange(0.000001, 0.00002) = 0.000001;
        lvq_epochs.setRange(5000, 10000) = 5000;
        lvq_seed.setRange(0, 10000) = 0;
        
        kernel_spreading.setRange(1, 2) = 1;
        gamma_spreading.setRange(10, 30) = 10;
        neighbors_spreading.setRange(1, 100) = 1;
        alpha_spreading.setRange(0, 1.0) = 0.0;
        epochs_spreading.setRange(20, 100) = 20;
    
        kernel_propagation.setRange(1, 2) = 1;
        gamma_propagation.setRange(10, 30) = 10;
        neighbors_propagation.setRange(1, 100) = 1;
        epochs_propagation.setRange(20, 100) = 20;

        //Add parameters to latin hypercube sampling:
        addParameterToLHS(a_t);
        addParameterToLHS(lp);
        addParameterToLHS(dsbeta);
        addParameterToLHS(age_wins);
        addParameterToLHS(e_b);
        addParameterToLHS(e_b_sup);
        addParameterToLHS(e_n);
        addParameterToLHS(epsilon_ds);
        addParameterToLHS(minwd);
        addParameterToLHS(epochs);
        addParameterToLHS(gamma);
        addParameterToLHS(h_threshold);
        addParameterToLHS(tau);
        addParameterToLHS(pushRate);
        addParameterToLHS(seed);
        
        addParameterToLHS(c);
        addParameterToLHS(kernel);
        addParameterToLHS(degree);
        
        addParameterToLHS(neurons);
        addParameterToLHS(hidden_layers);
        addParameterToLHS(lr);
        addParameterToLHS(momentum);
        addParameterToLHS(mlp_epochs);
        addParameterToLHS(activation);
        addParameterToLHS(lr_decay);
        addParameterToLHS(solver);

        addParameterToLHS(nnodes);
        addParameterToLHS(at_p);
        addParameterToLHS(at_n);
        addParameterToLHS(at_w);
        addParameterToLHS(lvq_tau);
        addParameterToLHS(lvq_epochs);
        addParameterToLHS(lvq_seed);
        
        
        addParameterToLHS(kernel_spreading);
        addParameterToLHS(gamma_spreading);
        addParameterToLHS(neighbors_spreading);
        addParameterToLHS(alpha_spreading);
        addParameterToLHS(epochs_spreading);
    
        //Label Prop
        addParameterToLHS(kernel_propagation);
        addParameterToLHS(gamma_propagation);
        addParameterToLHS(neighbors_propagation);
        addParameterToLHS(epochs_propagation);
    }
};

#endif /* MYPARAMETERS_H */
