/* 
 * File:   LVQNeuron.h
 * Author: hans
 *
 * Created on 28 de Fevereiro de 2012, 21:13
 */

#ifndef LVQNEURON_H
#define	LVQNEURON_H

#include "DSNeuron.h"
#include "Defines.h" 

class LVQNeuron : public DSNeuron {
private:
    int dClass;
    bool won;
       
            
public:
    MatVector<float> avgDistanceN; //average of distance of variable non-class
    
    LVQNeuron(){
        won = false;
        dClass = -1;
    };
    
    LVQNeuron(uint r, uint c) : DSNeuron(r, c) {
        won = false;
        dClass = -1;
    };
    
    LVQNeuron(uint r, uint c, int dClass) : DSNeuron(r, c) {
        setClass(dClass);
        won = false;
    };
    
    void setupWeights(uint inputSize);
    
    void setClass(int dClass);
    int getClass();    

    void setWon(bool won);
    bool getWon();
};

#endif	/* LVQNEURON_H */

