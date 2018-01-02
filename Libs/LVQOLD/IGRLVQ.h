/* 
 * File:   IGRLVQ.h
 * Author: flavia
 * Algoritmo iGRLVQ baseado no artigo: Incremental GRLVQ: Learning relevant features for 3D object recognition
 *                                     Kietzmann et al. 2008. Neurocomputing, 71 (2008) 2868-2879.
 *                                     Com modificações na insercao dos nodos e adição de funcao de remocao dos nodos
 * Created on 21 de Setembro de 2016, 15:40
 */


#ifndef IGRLVQ_H
#define	IGRLVQ_H

#include "LVQ.h"
#include <map>
#include "DebugOut.h"
#include "NodeW.h"
#include "LVQ21.h"
#include "GLVQ.h"
#include "GRLVQ.h" 

#define distWeight(x,y)   distEuclidianaWeight((x),(y))
#define distDeriv(x,y,i)  distWeightDerivV((x),(y),(i))
#define distDerivP(x,y,i) distDerivParam((x),(y),(i))

class IGRLVQ: public GRLVQ {

public:

//    TNumber training_error0;
//    TNumber training_error1;
//    TNumber dWMax0;
//    TVector vMax0;
//    TNumber clsMax0;
//    TNumber dWMax1;
//    TVector vMax1;
//    TNumber clsMax1;
    
    TNumber thrRemoveNode; //Limite minimo para remocao do nodo
    TNumber thrInsereNode; //Limite minimo para inserir do nodo
    
    TNumber insertNodeStart;
    TNumber removeNodeStart;
    
    TNumber id; //Node id
    
    TNumber countProt0; //Contador do numero de prototipos da classe 0
    TNumber countProt1; //Contador do numero de prototipos da classe 1
    
    
//    TNumber relSim;
    
    IGRLVQ(){
    }
    
    virtual ~IGRLVQ(){};
    
    //Hans: set max weights
    void setMaxWeights(TVector &maxWeight) {
        this->maxWeight = maxWeight;
    }

    //Hans: set max weights
    void setMinWeights(TVector &minWeight) {
        this->minWeight = minWeight;
    }
        
    //Inicialização com valores aleatorios
    virtual IGRLVQ& initialize(int nNodes, int ncls, int wsize){

        GRLVQ::initialize(nNodes, ncls, wsize);
        id = nNodes; //id
        countProt1 = 1;
        countProt0 = 1;
        
        return *this;
    }
    
    virtual IGRLVQ& insertNode(int cls, TVector v){
        
        LVQNode* node = new LVQNode(id, v, cls);
        node->winCount = 100;
        node->loseCount = 0;
        node->totalchange = 0;
        aloc_node++;
        meshNodeSet.insert(node);

        if(cls == 0){
            countProt0 =  countProt0 + 1;
        }
        else{
            countProt1 = countProt1 + 1;
        }
        id++;
        
        return *this;
    }
    
    void imprimeNode(LVQNode* node){
        
        dbgOut(0) << node->getId() << ": ";
        for(int i=0; i< node->w.size(); i++){
            dbgOut(0) << node->w[i] << " ";
        }
        dbgOut(0) << endl;
    }
    
   
//    IGRLVQ& trainning(int N = 1) {
//        int i = 0;
//        int tMaxTrainingError = data.rows()*0.1;
//        
//        do{
//            
////          Training iGRLVQ
//            total_change = 0;
//            total_changeW = 0;
//            training_error0 = 0;
//            training_error1 = 0;
//            
////            if(i==0){
////                
////                for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end();) {
////                    dbgOut(0) << "Nodo " << (*it)->getId() << ":\t";
////                    for(int i=0; i<5; i++){
////                    dbgOut(0) << (*it)->w[i] << "\t";
////                    }
////                    dbgOut(0) << "\tClasse:" << (*it)->cls << endl;
////                it++;
////                }
////                dbgOut(0) << "Weight:\t";
////                for (int i = 0; i < 5; i++) {
////                    dbgOut(0) << weight[i] << "\t";
////                }
////                dbgOut(0) << endl;
////            }
//            
//            
//            
//            
//            trainningStep();
//
//           
////          Remove e reseta os nodos perdedores
//            if (i > 50){
//                TPNodeSet::iterator it;
//                TNode *remove;
//                for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end();) {
//                    dbgOut(2) << "Nodo: " << (*it)->getId() << "\tVitorias: " << (*it)->winCount << "\tDerrotas: " << (*it)->loseCount << "\tTotal: " << meshNodeSet.size()<< endl;
//                    if((*it)->winCount <= thrRemoveNode && Mesh<TNode>::meshNodeSet.size() > 2){
//                        remove = (*it);
//                        dbgOut(2) << "Remove Nodo: " << "(" << (*it)->totalchange << ") ";
//                        //imprimeNode(remove);
//                        dbgOut(2) << "Remove Nodo: " << "(" << (*it)->totalchange << ") " << remove->getId() << "\tClasse: " << remove->cls << "\tVitorias: " << (*it)->winCount << "\tDerrotas: " << (*it)->loseCount << "\t Ciclo: " << i <<  endl;
//                        it++;
//                        meshNodeSet.erase(remove);
//                    }
//                    else {//Se o nodo não foi excluido, eh entao resetado e o ponteiro vai para o proximo nodo
//                        (*it)->winCount = (*it)->winCount-30;
//                        //if((*it)->winCount < 0) (*it)->winCount = 0;
//                        //(*it)->winCount = 0;
//                        (*it)->loseCount = 0;
//                        it++;
//                    }                                
//                }
//            }
//            
//            
////            Insere um novo nodo para o acumulado de erro 
//            if (i > 20 && i< N/2){
//                TPNodeSet::iterator it;
//                TNode *add;
//                add = NULL;
//                
//                for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end();) {
//                    if(add == NULL){
////                        if((*it)->loseCount > data.rows()*0.02){
//                        if((*it)->loseCount > thrInsereNode){    
//                            (*it)->loseCount = 0;
//                            add = (*it);
//                        }
//                    }
//                    else{
//                        if((*it)->loseCount > add->loseCount){
//                            (*it)->loseCount = 0;
//                            add = (*it);
//                        }
//                    }
//                    (*it)->loseCount = 0;
//                    it++;
//                }
//                if(add != NULL){
//                    if(add->cls == 0){
//                        insertNode(1, add->w.size(), add->w);
//                    }
//                    else{
//                        insertNode(0, add->w);
//                    }
//                }
//            }
//           
////            Insere um novo nodo quando ultrapassar o erro maximo de cada classe
////            if (i > 30 && i< N/2 && training_error0 > tMaxTrainingError){
////                TVector v(vMax0.size()); //cria um vetor para o nodo
////                v.random().mult(0.1).add(0.45); //aleatoriza em torno de 0.5               
////                insertNode(0, v); //j classe 0              
//////                insertNode(clsMax0, vMax0.div(training_error0));
//////                insertNode(clsMax0, vMax0.size(), vMax0);
////                dbgOut(2) << "Training Error 0: " << training_error0 << "\t" << "Nodos: " << meshNodeSet.size() << endl;
////            }
////            
////            if (i > 30 && i< N/2 && training_error1 > tMaxTrainingError){               
////                TVector v(vMax0.size()); //cria um vetor para o nodo
////                v.random().mult(0.1).add(0.45); //aleatoriza em torno de 0.5
////                insertNode(1, v); //j classe 1
//////                insertNode(clsMax1, vMax1.div(training_error1));
//////                insertNode(clsMax1, vMax1);
////                dbgOut(2) << "Training Error 1: " << training_error1 << "\t" << "Nodos: " << meshNodeSet.size() << endl;
////            }
//            total_changeW = total_changeW/data.rows();
//            total_change = total_change/data.rows();
//            
////            if (i%100==0){
////                dbgOut(0) << i << "\tC:\t" <<total_change << "\tCw:\t" << total_changeW << "\t";
////                for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end();it++) {
////                    
////                    dbgOut(2) << (*it)->cls << ":\t" << average((*it)->w, 0, 4) << "\t" << average((*it)->w, 5, (*it)->w.size()-1) << "\t";
////                }
////                dbgOut(0) << average(weight, 0, 4) << "\t" << average(weight, 5, weight.size()-1) << endl;
////            }
//            i++;
//        } while (i < N && total_change > min_change);
//        
//        dbgOut(0) << "Ciclos: " << i << "\tChange: " << total_change << "\tChangeW: " << total_changeW << endl;
//        
//        
//        //Calcula a Acuracia Geral
//        TPNodeSet::iterator it;
//        TNumber tp = 0;
//        
//        for(int i=0; i<data.rows(); i++){
//            TNode* winner; 
//            TNode* runnerUp;
//            TVector row;
//            TNumber dR; 
//            TNumber dW;
//            
//            data.getRow(i, row);
//            getWinnerRunnerUpWeight(row, winner, runnerUp, vcls[i]);
//            
//            dR = distWeight(row, *runnerUp);
//            dW = distWeight(row, *winner);
//            
//            if(dW < dR){
//                tp++;
//            }
//        }
//        dbgOut(0) << "Acuracia: " << tp/data.rows() << endl;
//        return *this;
//    }
    
    IGRLVQ& trainning(int N = 1) {
        TPNodeSet::iterator it;
        TNode *remove;
        TNode *add;
        int i = 0;

        do{
            
//          Training iGRLVQ
            total_change = 0;
            total_changeW = 0;

            trainningStep();
            
//          Remove e reseta os nodos perdedores
            if (i > removeNodeStart && i< N/2){

                for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end(); it++) {
                    
                    if((*it)->winCount <= thrRemoveNode && Mesh<TNode>::meshNodeSet.size() > 2){
                        if((*it)->cls == 0 && countProt0 > 1){
                            remove = (*it);
                            meshNodeSet.erase(remove);
                            countProt0 = countProt0 - 1;
                        }
                        else if ((*it)->cls == 1 && countProt1 > 1){
                            remove = (*it);
                            meshNodeSet.erase(remove);
                            countProt1 = countProt1 - 1;
                        }
                    }
                }
            }
            
            
//            Insere um novo nodo para o acumulado de erro 
            if (i > insertNodeStart && i< N/2){
                add = NULL;
                
                for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end(); it++) {
                    
                    if((*it)->loseCount > thrInsereNode && (add == NULL || ((*it)->loseCount > add->loseCount))){    
                        add = (*it);
                        
                    }
                }
                if(add != NULL){
                    dbgOut(2) << add->loseCount << "/"<< meshNodeSet.size() << endl;
                    if(add->cls == 0){
                        insertNode(1, add->w);
                    }
                    else{
                        insertNode(0, add->w);
                    }
                }
            }

            for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end(); it++) {
                (*it)->loseCount = 0;
                (*it)->winCount = 0;
            }
            
            total_changeW = total_changeW/data.rows();
            total_change = total_change/data.rows();
            
            if (i%100==0){
                
                dbgOut(1) << i << "\t" <<total_change << "\t" << total_changeW << "\t";
                for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end();it++) {
                    
                    dbgOut(1) << (*it)->cls << ":\t" << average((*it)->w, 0, 4) << "\t" << average((*it)->w, 5, (*it)->w.size()-1) << "\t";
                }
                dbgOut(1) << average(weight, 0, 4) << "\t" << average(weight, 5, weight.size()-1) << endl;
            }
            i++;
        } while (i < N && total_changeW > min_change); //(total_changeW > min_change);
        
        dbgOut(0) << "Ciclos: " << i << "\tChange: " << total_change << "\tChangeW: " << total_changeW << endl;
        
        
        //Calcula a Acuracia Geral
        TNumber tp = 0;
        
        for(int i=0; i<data.rows(); i++){
            TNode* winner; 
            TNode* runnerUp;
            TVector row;
            TNumber dR; 
            TNumber dW;
            
            data.getRow(i, row);
            getWinnerRunnerUpWeight(row, winner, runnerUp, vcls[i]);
            
            dR = distWeight(row, *runnerUp);
            dW = distWeight(row, *winner);
            
            if(dW < dR){
                tp++;
            }
        }
        dbgOut(0) << "Acuracia: " << tp/data.rows() << endl;
        return *this;
    }
    
    
    virtual IGRLVQ& trainningStep() {
        
//        dWMax0 = 10000; //distancia maxima da amostra ao nodo da classe 0
//        dWMax1 = 10000; //distancia maxima da amostra ao nodo da classe 1
//        relSim = 1;
        for(int i=0; i < data.rows(); i++){
            TVector v;
            int row = rand()%data.rows();
            //std::cout << "Row: " << row << std::endl;
            data.getRow(row, v);
            updateMap(v, vcls[row]);        
        }
//        dbgOut(2) << "relSim: " << relSim << endl;
        return *this;
    }
    
    virtual IGRLVQ& trainningEach(int N = 1) {
        MatVector<int> vindex(data.rows());
        vindex.range(0, vindex.size() - 1);
        
        int vSize=vindex.size();
        vindex.srandom(MatUtils::getCurrentCPUTimer());
        vindex.shuffler();
        
        TVector v;
        for (int n = 0; n < N; n++) {
            total_change = 0;            
            for (int l = 0; l < vindex.size(); l++) {
                data.getRow(vindex[l], v);
                updateMap(v, vcls[vindex[l]]);
            }
            dbgOut(0) << "It: " << n << "\t" << "Change Proto: " << total_change << endl;
        }       

        return *this;
    }

    
    float average(const TVector &w, int start, int end){
        float soma = 0;
        
        for (int i = start; i <= end; i++) {
            soma = soma + w[i];
        }
        return soma/(end-start+1);
    }
    
    
    
    virtual IGRLVQ& updateMap(const TVector &w, int cls) {
        TNode *winner, *runnerUp;
                
        //Pega os prototipos mais proximos de classes distintas
        getWinnerRunnerUpWeight(w, winner, runnerUp, cls);
        
        //Variavel que calcula o quanto o prototipo modificou
        TNumber proto_changeRunner = 0; 
        TNumber proto_changeWinner = 0;

        //Calcula as distancias com pesos da amostra para ambos os prototipos
        if(winner != NULL && runnerUp != NULL){
            TNumber dR, dW;
            dR = distWeight(w, *runnerUp);
            dW = distWeight(w, *winner);

//            dbgOut(2) << "dR:" << dR << "\tdW:" << dW << endl;
            
            if(fabs(dR) > 10e-10 && fabs(dW) > 10e-10){
                //Calcula passo a passo
                TNumber sqsum, xW, xR, sgd, factorW, factorR;
                sqsum = (dW + dR) * (dW + dR);
                xW = 2 * dR / sqsum; 
                xR = 2 * dW / sqsum;
                sgd = sgdprime((dW - dR) / (dW + dR)); //Calcula sigmoide
                factorW = alpha_tp * sgd * xW; 
                factorR = - alpha_tn * sgd * xR; 
                        
                for (int i = 0; i < w.size(); i++) {
                    TNumber updateW, updateR;
                    updateW = factorW * (distDeriv(w, *winner, i));
                    updateR = factorR * (distDeriv(w, *runnerUp, i));

                    //Atualiza os prototipos Winner e Runnerup
                    if(updateW && updateR == 0) continue;
                    winner->w[i] = winner->w[i] + updateW;
                    runnerUp->w[i] = runnerUp->w[i] + updateR;

                    //Soma as mudanças dos prototipos
                    //proto_changeRunner += updateR * updateR;
                    //proto_changeWinner += updateW * updateW;
                }

                updateDistanceMeasure(w, *winner, *runnerUp);

                //Atualiza o total de mudanças ocorridas nos prototipos
                //total_change += (sqrt(proto_changeRunner) + sqrt(proto_changeWinner));
                //dbgOut(1) << proto_changeRunner << "\t" << proto_changeWinner << endl;
                //winner->totalchange = winner->totalchange + sqrt(proto_changeWinner);
                //runnerUp->totalchange = runnerUp->totalchange + sqrt(proto_changeRunner);
                //Calcula o erro de classificacao
                if (dW > dR) { //O Nodo Runner esta mais proximo da amostra que o Winner
                    //Guarda a amostra com a menor distância para cada classe
                    runnerUp->loseCount++;
                }
            }
        }
        int count = 0;
        for (int i = 0; i < weight.size(); i++) {
            if(weight[i] == 0){
                count++;
            }
        }
        dbgOut(0) << "Quantidade de zeros: " << count << endl;
        learningDecay(); // Faz o decaimento dos valores dos parâmetros

    }
    
};

#endif	/* IGRLVQ_H */

