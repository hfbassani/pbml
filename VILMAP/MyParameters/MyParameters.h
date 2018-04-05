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

    MyParameters() {
        comments = "Test float Parameters";
        section = "VILMAP Parameters";

        //make parameters persistent:
        addParameterD(N, "Number of samples");
        addParameterD(a_t, "Activation");
        addParameterD(lp, "Cluster Percentage");
        addParameterD(dsbeta, "Relevance rate");
        addParameterD(age_wins, "Max Competitions");
        addParameterD(e_b, "learning rate");
        addParameterD(e_n, "Cluster Percentage");
        addParameterD(epsilon_ds, "Relevance rate");
        addParameterD(minwd, "Relevance rate");

        //Set default ranges and values
        N = 500;
        a_t.setRange(0.70, 0.99) = 0.7;
        lp.setRange(0.01, 0.1) = 0.01;
        dsbeta.setRange(0.001, 0.1) = 0.01;
        age_wins.setRange(1, 100) = 1;
        e_b.setRange(0.001, 0.1) = 0.01;
        e_n.setRange(0.0001, 0.5) = 0.001;
        epsilon_ds.setRange(0.01, 0.1) = 0.01;
        minwd.setRange(0, 0.5) = 0.05;

        //Add parameters to latin hypercube sampling:
        addParameterToLHS(a_t);
        addParameterToLHS(lp);
        addParameterToLHS(dsbeta);
        addParameterToLHS(age_wins);
        addParameterToLHS(e_b);
        addParameterToLHS(e_n);
        addParameterToLHS(epsilon_ds);
        addParameterToLHS(minwd);
    }

};

#endif /* MYPARAMETERS_H */
