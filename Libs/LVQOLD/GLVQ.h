/* 
 * File:   GLVQ.h
 * Author: flavia
 *
 * Created on 30 de Abril de 2014, 17:17
 * 
 * Algoritmo baseado na versão da GLVQ proposta por Sato e Yamada, 1996.
 * Generalized learning vector quantization.
 * 
 */

#ifndef GLVQ_H
#define	GLVQ_H

#include "LVQ.h"
#include <map>
#include "DebugOut.h"
#include "NodeW.h"
#include "LVQ21.h"


class GLVQ: public LVQ<LVQNode> {
    
public:
    TNumber window;     //largura da janela
    TNumber alpha;      //taxa de aprendizagem inicial
    TNumber alpha_tp;    //taxa de aprendizagem com decaimento positivos
    TNumber alpha_tn;    //taxa de aprendizagem com decaimento negativos
    TNumber tau;        //taxa de decaimento
    TNumber t;          //Numero de epochs
    TNumber tmax;       //Numero maximo de epochs
    
    GLVQ(){
    }
    
    virtual ~GLVQ(){};
    
    virtual GLVQ& initialize(int nNodes, int ncls, int wsize){
        t = 0; //inicializa count do numero de epochs
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
    
    virtual TNumber sgd(TNumber x) {
            return 1/(1 + exp(-x));
    }

    /**
     * First derivative of the logisic function sgd(x).
     * 
     * sgd(x)  = 1/(1 + exp(-x))
     * sgd'(x) = sgd(x) * (1 - sgd(x))
     * 
     * @param x
     * @return
     */
    
    virtual TNumber sgdprime(TNumber x) {
        TNumber s = sgd(x);
        return s * (1-s);
    }
    
    
    virtual TNumber distance(const TVector &w, const TNode &node, int i){
        //Calcula a distancia derivV
        return 2 * (w[i] - node.w[i]);
    }
        
    virtual GLVQ& updateMap(const TVector &w, int cls) {
        TNode *winner, *runnerUp;
        
        //Pega os prototipos mais proximos de classes distintas
        getWinnerRunnerUp(w, winner, runnerUp, cls);
        
        //Calcula as distancias da amostra para ambos os prototipos
        if(winner != NULL && runnerUp != NULL){
            TNumber dR, dW;
            dR = w.dist(runnerUp->w);
            dW = w.dist(winner->w);
            dbgOut(2) << "dRunnerUp: " << dR << endl;
            dbgOut(2) << "dWinner: " << dW << endl;
            
            //Calcula 
            TNumber sqsum, xW, xR, sgd, factorW, factorR;
            sqsum = (dW + dR) * (dW + dR);
            xW = 2 * dR / sqsum; 
            xR = 2 * dW / sqsum;
            sgd = sgdprime((dW - dR) / (dW + dR)); //Calcula sigmoide
            factorW = alpha_tp * sgd * xW;
            factorR = - alpha_tn * sgd * xR;
            
            for (int i = 0; i < w.size(); i++) {
                TNumber updateW, updateR;
                updateW = factorW * (distance(w, *winner, i));
                updateR = factorR * (distance(w, *runnerUp, i));
                
                //Atualiza os prototipos Winner e Runnerup
                //TODO VERIFICAR SE ESSES SINAIS ESTAO REALMENTE TROCADOS
                winner->w[i] = winner->w[i] + updateW;
                runnerUp->w[i] = runnerUp->w[i] + updateR;
            }
            dR = w.dist(runnerUp->w);
            dW = w.dist(winner->w);
            dbgOut(2) << "dRunnerUp: " << dR << endl;
            dbgOut(2) << "dWinner: " << dW << endl;
        }
    }
    
    virtual GLVQ& trainningStep(const TVector &w, int cls){
        updateMap(w, cls);
    }
    
    virtual GLVQ& learningDecay(){
        
        if (t < tmax) {
            t++;
            alpha_tp = (alpha_tp)/(1 + tau * (t - 0)); //Equacao (Nova e Estevez, 2013 Neural Comput&Applic)
            alpha_tn = (alpha_tn)/(1 + tau * (t - 0));
        }
    }

};

#endif	/* GLVQ_H */

