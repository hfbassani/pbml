/* 
 * File:   SRNG.h
 * Author: flavia
 *
 * Created on 8 de Março de 2016, 10:42
 */
#include "GRLVQ.h"
#include <map>
#include "DebugOut.h"
#include "NodeW.h"

#ifndef SRNG_H
#define	SRNG_H

#define LOG_NODE 18

class SRNG: public GRLVQ {
public:
    TNumber gamma;       //cooperatividade da vizinhanca
    TNumber h_threshold;
    
    TVector h;
    std::vector<LVQNode*> ranks;
    LVQNode* runnerUp;
    
    virtual SRNG& initialize(int nNodes, int ncls, int wsize) {
        GRLVQ::initialize(nNodes, ncls, wsize);
               
        //Precompute h
        h.size(nNodes);
        for(int i=0;i<nNodes;i++) {
            h[i] = exp( - ((double) i / gamma));
//            dbgOut(0) << "h[" << i << "]: " << h[i] << endl;
        }
        return *this;
    }
    
    double calcC(int n) {
       TNumber C=0;
       for(int i=0;i<n;i++) {
            C += h[i];
       }
       return C;
    }
    
    void insertInRank(LVQNode *node) {
    
        std::vector<LVQNode*>::iterator it;
        for (it=ranks.begin(); it != ranks.end(); it++) {
            LVQNode* r = (*it);
            if ( node->dist < r->dist ) break;
        }
        
        ranks.insert(it, node);
    }
    
    virtual LVQNode* updateDistsRankRunnerUp(const TVector &w, int cls) {
        ranks.clear();
        runnerUp = NULL;
        
        typename TPNodeSet::iterator it;
        for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end(); it++) {
            LVQNode* node = (*it);
            
            node->dist = distWeight(w, *node);
            //node->dist = node->w.dist(w);
            //node->dist = distEuclidianaWeight(w, *node);
            
            //dbgOut(0) << "Node[" << node->getId() << "]->dist:" << node->dist << endl;
//            if(node->dist < 0.000){
//                    dbgOut(0) << "HL: " << h.toString() << endl;
//                    dbgOut(0) << "weight: " <<  weight.toString() << endl;
//                    dbgOut(0) << "w: " << w.toString() << endl;
//                    dbgOut(0) << "Winner.w: " << node->w.toString() << endl;
//            }
            
            if (node->cls == cls) {
                insertInRank(node);
            } else {
                if (runnerUp == NULL || runnerUp->dist > node->dist)
                    runnerUp = node;
            }
        }
        //printNodeLog(ranks[0], w, cls, "updateDistsRankRunnerUp");
    }
    
    void printRankAndRunnerUp() {
        int r = 0;
        dbgOut(0) << "Nodes in rank: " << endl;
        for (std::vector<LVQNode*>::iterator it = ranks.begin(); it != ranks.end(); it++) {
            LVQNode* node = (*it);
            dbgOut(0) << r << " - Node(" << node->getId() << ") - cls: " << node->cls << endl; 
            r++;
        }
        dbgOut(0) << "RunnerUp: Node(" << runnerUp->getId() << ") - cls: " << runnerUp->cls << endl; 
    }
    
    bool printNodeLog(LVQNode *node, const TVector &w, int cls, const std::string str) {
        if (node->getId() == LOG_NODE) {
            dbgOut(0) << str << "[t=" << t << "]:" << endl;
            dbgOut(0) << "winnerRk[" << node->getId() <<"].w: " << node->w.toString() << endl;
            dbgOut(0) << "RunnerUp[" << runnerUp->getId() <<"].w: " << runnerUp->w.toString() << endl;
            dbgOut(0) << "weight: " <<  weight.toString() << endl;
            dbgOut(0) << "w: " << w.toString() << endl << endl;
            return true;
        } 
        return false;
    }
    
    virtual SRNG& updateMap(const TVector &w, int cls) {
        TNumber total_change = 0;
        updateDistsRankRunnerUp(w, cls);
        
        double C = calcC(ranks.size());
        TNumber dR, dW, sum, s, sqsum, xW, xR, sgd, factorW, factorR;
        int r; //rank
        dR = runnerUp->dist;
        LVQNode* winnerRk;
        sum = r = 0;
        
        //Atualiza todos os nodos da mesma classe do padrao
        TNumber changeWinner = 0;
        for (std::vector<LVQNode*>::iterator it = ranks.begin(); it != ranks.end(); it++) {
            winnerRk = (*it);
            
            //Calcula passo a passo            
            dW = winnerRk->dist;
            
            sqsum = (dW + dR) * (dW + dR);
            xW = 2 * dR / sqsum; 
            xR = 2 * dW / sqsum;
            sgd = sgdprime((dW - dR) / (dW + dR)); //Calcula sigmoide
            s = sgd * xW * h[r] / C;
            sum += s;
            factorW = alpha_tp * s;

//            if (xW+xR>500) {
//                dbgOut(0) << "C:" << C << " h[" << r << "]:" << h[r] << " dW:" << dW << " dR:" << dR << " xW:" << xW << " xR:" << xR << endl; 
//                if (r==0)
//                   dbgOut(0) << "RunnerUp[" << runnerUp->getId() <<"].w: " << runnerUp->w.toString() << endl;
//                dbgOut(0) << "winnerRk[" << winnerRk->getId() <<"].w: " << winnerRk->w.toString() << endl;
//                dbgOut(0) << "s:" << s << " sum:" << sum << endl; 
//            }
            
            
            for (int i = 0; i < w.size(); i++) {
                //TNumber updateW = factorW * (distDeriv(w, *winnerRk, i));
                TNumber updateW = factorW * (distWeightDerivV(w, *winnerRk, i));
                winnerRk->w[i] = winnerRk->w[i] + updateW;
                changeWinner += updateW * updateW;
            }
            
            //Atualiza a distancia antes de calcular a atualizacao da metrica
            //Obs.: Diferente do codigo do libgrlvq
            //winnerRk->dist = distEuclidianaWeight(w, *winnerRk);
            
            //printNodeLog(winnerRk, w, cls, "updateMap:");
            r++;
            if (h[r]<h_threshold) break; // nao prossegue com a atualizacao dos vizinhos se h for muito pequeno
        }
        
        //Atualiza o nodo da classe contraria ao do padrao
        factorR = -alpha_tn * sum; //Invertendo os sinais
        TNumber changeRunner = 0; 
        for (int i = 0; i < w.size(); i++) {
            //TNumber updateR = factorR * (distDeriv(w, *runnerUp, i));
            TNumber updateR = factorR * (distWeightDerivV(w, *runnerUp, i));
            runnerUp->w[i] = runnerUp->w[i] + updateR;
            changeRunner += updateR * updateR;
        }
        //Atualiza a distancia antes de calcular a atualizacao da metrica
        //Obs.: Diferente do codigo do libgrlvq
        //runnerUp->dist = distEuclidianaWeight(w, *runnerUp);
        total_change += sqrt(changeRunner) + sqrt(changeWinner);
        updateDistanceMeasure(w, C);
        learningDecay();
        dbgOut(2) << "Change: " << total_change << endl;
        return *this;
    }

    virtual SRNG& updateDistanceMeasure(const TVector &w, double C){
        
        TNumber dR, dW, sqsum, xW, xR, sgd, factorW;
        TVector deltaW(w.size());
        deltaW.fill(0);
        int r;
        dR = runnerUp->dist;
        LVQNode* winnerRk;

        //Calculo da relevancia
        
        r = 0;
        for (std::vector<LVQNode*>::iterator it = ranks.begin(); it != ranks.end(); it++) {  
            winnerRk = (*it);
            dW = winnerRk->dist;
            sqsum = (dW + dR) * (dW + dR);
            xW = 2 * dR / sqsum; 
            xR = 2 * dW / sqsum;
            sgd = sgdprime((dW - dR) / (dW + dR)); //Calcula sigmoide
            factorW = -alpha_w * sgd * h[r] / C;
            
            for (int i = 0; i < w.size(); i++) {
                //deltaW[i] += factorW * ((xW * (distDerivP(w, *winnerRk, i))) - (xR * (distDerivP(w, *runnerUp, i))));
                deltaW[i] += factorW * ((xW * (distDerivParam(w, *winnerRk, i))) - (xR * (distDerivParam(w, *runnerUp, i))));
            }
            
            //if(deltaW[0]<-0.5){
            //    dbgOut(0) << deltaW.toString() << endl;  
            //}
            
            //printNodeLog(winnerRk, w, 0, "updateDistanceMeasure");
            
            r++;
            if (h[r]<h_threshold) break; // nao prossegue com a atualizacao dos vizinhos se h for muito pequeno
        }
        updateWeightVector(deltaW);
    }

    
//    virtual SRNG& updateDistanceMeasure(const TVector &w, double C){
//     
//        TNumber dR, dW, sqsum, xW, xR, sgd, sum, s;
//        TVector deltaW(w.size());
//        int r;
//        dR = runnerUp->dist;
//        LVQNode* winnerRk;
//
//        //Calculo da relevancia
//        for (int i = 0; i < w.size(); i++) {
//            sum = r = 0;
//            for (std::vector<LVQNode*>::iterator it = ranks.begin(); it != ranks.end(); it++) {  
//                winnerRk = (*it);
//                dW = winnerRk->dist;
//                sqsum = (dW + dR) * (dW + dR);
//                xW = 2 * dR / sqsum; 
//                xR = 2 * dW / sqsum;
//                sgd = sgdprime((dW - dR) / (dW + dR)); //Calcula sigmoide
//                
//                sum += sgd * h[r] / C * ((xW * (distDerivParam(w, *winnerRk, i))) - (xR * (distDerivParam(w, *runnerUp, i))));
//                r++;
//                if (h[r]<h_threshold) break; // nao prossegue com a atualizacao dos vizinhos se h for muito pequeno
//            }
//            deltaW[i] = - alpha_w * sum;
//        }
//        updateWeightVector(deltaW);
//    }
    
};

#endif	/* SRNG_H */

