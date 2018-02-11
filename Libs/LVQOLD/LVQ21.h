/* 
 * File:   LVQ21.h
 * Author: flavia
 *
 * Created on 16 de Abril de 2014, 10:27
 * 
 * Algoritmo baseado na versão da LVQ21.h de Kohonen, 1990.
 * Improved Versions of Learning Vector Quantization
 * 
 */

#ifndef LVQ21_H
#define	LVQ21_H

#include "LVQ.h"
#include <map>
#include "DebugOut.h"
#include "NodeW.h"

class LVQNode: public NodeW {
public:
typedef std::map<LVQNode*, TConnection*> TPNodeConnectionMap;
TPNodeConnectionMap nodeMap;

    int cls;
    float dist;
    
    LVQNode(int idIn, const TVector &v, const int c) : NodeW(idIn, v), cls(c) {
    };
    
    LVQNode(int idIn, const TVector &v) : NodeW(idIn, v), cls(0) {
    };
    
    void write(std::ofstream &file) {
        
        for (int i=0; i<w.size(); i++) {
            file << w[i];
            if (i<w.size()-1) 
                file << "\t";
        }
        file << std::endl << cls << std::endl;
    }
    
    void read(std::istream &file) {
        
        std::string line;
        getline(file, line);
        std::stringstream parserW(line);
        float value;
        int i=0;
        while (!parserW.eof()) {
            parserW >> value;
            if (i<w.size())
                w[i] = value;
            else
                w.append(value);
            i++;
        }
        getline(file, line);
        std::stringstream parserCLS(line);
        parserCLS >> cls;
    }    
};

class LVQ21: public LVQ<LVQNode> {
    
public:
    TNumber window;     //largura da janela
    TNumber alpha;      //taxa de aprendizagem inicial
    TNumber alpha_t;    //taxa de aprendizagem com decaimento
    TNumber tau;        //taxa de decaimento
    TNumber t;          //Numero de epochs
    TNumber tmax;       //Numero maximo de epochs
    
    LVQ21(){
    }
    
    virtual ~LVQ21(){};
    
    virtual LVQ21& initialize(int nNodes, int ncls, int wsize){
        t = 0; //inicializa o numero de epochs
        alpha_t = alpha;
        int cls = 0;

        for (int x = 0; x < nNodes; x++){
            TVector v(wsize); //cria um vetor
            v.random(); //aleatoriza

            LVQNode* node = new LVQNode(x, v, cls);
            aloc_node++;
            meshNodeSet.insert(node);

            cls++;
            
            if (cls == ncls){
                cls = 0;
            }
        }
    }
    
    virtual LVQ21& updateMap(const TVector &w, int cls) {
        TNode *winner, *runnerUp;
        getWinnerRunnerUp(w, winner, runnerUp, cls);
        
        //Calcula as distancias da amostra para os prototipos
        if(winner != NULL && runnerUp != NULL){
            TNumber dR, dW;
            dR = w.dist(runnerUp->w);
            dW = w.dist(winner->w);
            
            dbgOut(2) << "dRunnerUp: " << dR << endl;
            dbgOut(2) << "dWinner: " << dW << endl;
            
            //Verifica se a amostra esta dentro da janela
            dbgOut(2) << "Minimo: " << std::min(dR/dW, dW/dR) << " > " << ((1-window)/(1+window)) << endl;
            if(std::min(dR/dW, dW/dR) > ((1-window)/(1+window))){
            //Atualiza o mapa
                winner->w = winner->w + (alpha_t * (w - winner->w));
                runnerUp->w = runnerUp->w - (alpha_t * (w - runnerUp->w));
                dbgOut(2) << "Mapa Atualizado" << endl;
                
            }
        }
    }
    
    virtual LVQ21& trainningStep(const TVector &w, int cls){
        updateMap(w, cls);
    }
    
    virtual LVQ21& learningDecay(){
        
        if (t < tmax) {
            t++;
            alpha_t = (alpha)/(1 + tau * (t - 0)); //Equacao (Nova e Estevez, 2013 Neural Comput&Applic)
        }
    }

};

#endif	/* LVQ21_H */

