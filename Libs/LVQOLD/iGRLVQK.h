/* 
 * File:   iGRLVQK.h
 * Author: flavia
 * Algoritmo iGRLVQ baseado no artigo: Incremental GRLVQ: Learning relevant features for 3D object recognition
 *                                     Kietzmann et al. 2008. Neurocomputing, 71 (2008) 2868-2879
 * Created on 21 de Setembro de 2016, 10:20
 */


#ifndef iGRLVQK_H
#define	iGRLVQK_H

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

class iGRLVQK: public GRLVQ {

public:

    TNumber training_error0;
    TNumber training_error1;
    TNumber dWMax0;
    TVector vMax0;
    TNumber clsMax0;
    TNumber dWMax1;
    TVector vMax1;
    TNumber clsMax1;
    
    TNumber thrRemoveNode; //Limite minimo para remocao do nodo


    
    iGRLVQK(){
    }
    
    virtual ~iGRLVQK(){};
    
    //Hans: set max weights
    void setMaxWeights(TVector &maxWeight) {
        this->maxWeight = maxWeight;
    }

    //Hans: set max weights
    void setMinWeights(TVector &minWeight) {
        this->minWeight = minWeight;
    }
        
    //Inicialização com valores aleatorios
    virtual iGRLVQK& initialize(int nNodes, int ncls, int wsize){

        GRLVQ::initialize(nNodes, ncls, wsize);
    }
    
    void insertNode(int cls, int wsize, TVector v){
        int x = meshNodeSet.size()+1;
        
        LVQNode* node = new LVQNode(x, v, cls);
        node->winCount = 100;
        aloc_node++;
        meshNodeSet.insert(node);
            
    }
    
   
    iGRLVQK& trainning(int N = 1) {
        int i = 0;
        int tMaxTrainingError = data.rows()*0.9;
        TPNodeSet::iterator it;
        
        do{
            total_change = 0;
//            training_error0 = 0;
//            training_error1 = 0;
            trainningStep();
            //dbgOut(2) << "i: " << i << "\t" << "Change Proto: " << total_change << "\t" << "Classification Error: " << training_error << "\t" << "Positiva: " << alpha_tp << "\t" << "Negativa: " << alpha_tn << "\t" << "Peso: " << alpha_w << endl;
            i++;
                        
            //Insere um novo nodo de acordo com o erro maximo de cada classe
            if (i > 30 && training_error0 > tMaxTrainingError){
                //TVector v(vMax0.size()); //cria um vetor para o nodo
                //v.random().mult(0.1).add(0.45); //aleatoriza em torno de 0.5               
                //insertNode(0, v.size(), v); //j classe 0
                insertNode(clsMax0, vMax0.size(), vMax0);
                dbgOut(2) << "Training Error 0: " << training_error0 << "\t" << "Nodos: " << meshNodeSet.size() << endl;
                training_error0 = 0;
            }
            
            if (i > 30 && training_error1 > tMaxTrainingError){
                //TVector v(vMax0.size()); //cria um vetor para o nodo
                //v.random().mult(0.1).add(0.45); //aleatoriza em torno de 0.5
                //insertNode(1, v.size(), v); //j classe 1
                insertNode(clsMax1, vMax1.size(), vMax1);
                dbgOut(2) << "Training Error 1: " << training_error1 << "\t" << "Nodos: " << meshNodeSet.size() << endl;
                training_error1 = 0;
            }
            
            dbgOut(2) << i << " : " << total_change << endl;
        } while (i < N && total_change > min_change);     
        return *this;
    }
    
    virtual iGRLVQK& trainningStep() {
        
        dWMax0 = 100000; //distancia maxima da amostra ao nodo da classe 0
        dWMax1 = 100000; //distancia maxima da amostra ao nodo da classe 1       
        for(int i=0; i < data.rows(); i++){
            TVector v;
            if (data.rows()>0) {
                int row = rand()%data.rows();
                //std::cout << "Row: " << row << std::endl;
                data.getRow(row, v);
                updateMap(v, vcls[row]);        
            }
        }
        return *this;
    }
    
    virtual iGRLVQK& trainningEach(int N = 1) {
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

    virtual iGRLVQK& updateMap(const TVector &w, int cls) {
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
            
            if(dR != 0 && dW != 0){
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
                    winner->w[i] = winner->w[i] + updateW;
                    runnerUp->w[i] = runnerUp->w[i] + updateR;

                    //Soma as mudanças dos prototipos
                    proto_changeRunner += updateR * updateR;
                    proto_changeWinner += updateW * updateW;
                }

                updateDistanceMeasure(w, *winner, *runnerUp);

                //Atualiza o total de mudanças ocorridas nos prototipos
                total_change += ((sqrt(proto_changeRunner) + sqrt(proto_changeWinner)))/meshNodeSet.size();

                //Calcula o erro de classificacao
                if (dW > dR) {
                    //Guarda a amostra com a maior distância para cada classe
                    if(cls == 0){
                        if (dR < dWMax0){ //Verifica se o dR e o menor possivel e armazena para insercao do nodo na posicao
                            dWMax0 = dW;
                            vMax0 = w;
                            clsMax0 = cls;
                        }
                        training_error0 = training_error0 + 1;
                    }
                    else{
                        if (dR < dWMax1){
                            dWMax1 = dW;
                            vMax1 = w;
                            clsMax1 = cls;
                        }
                        training_error1 = training_error1 + 1;
                    }
                }
            }
        }
//        else{
//            dbgOut(0) << "Winner: " << winner << "\t Runner: " << runnerUp << endl;
//        }
        learningDecay();
    }
    
};

#endif	/* iGRLVQK_H */

