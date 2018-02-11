/* 
 * File:   IGRLVQRD.h
 * Author: flavia
 * Algoritmo iGRLVQRD:  Neste algoritmo e feita um extesão do algoritmo IGRLVQ-SNPI, 
 *                      considerando a reducao da dimensionalidade dos dados. E criado
 *                      um novo vetor onde sao indicados as dimensoes dos dados que 
 *                      se mantem relevantes.
 * 
 * Created on 14 de Agosto de 2017, 11:24
 */


#ifndef IGRLVQRD_H
#define	IGRLVQRD_H

#include "GRLVQ.h" 
#include "IGRLVQ.h" 


class IGRLVQRD: public IGRLVQ {

public:

    TNumber thrRemoveNode; //Limite minimo para remocao do nodo
    TNumber thrInsereNode; //Limite minimo para inserir do nodo
    
    TNumber insertNodeStart;
    TNumber removeNodeStart;
    
    TNumber id; //Node id
    
    TNumber countProt0; //Contador do numero de prototipos da classe 0
    TNumber countProt1; //Contador do numero de prototipos da classe 1
    
    TVector vectorRelInc; // Vetor com incrementos das posicoes relevantes
    TVector vectorMaxRel; // Vetor que acumula os valores máximos obtidos por cada dimensao
    TNumber raizVectorRelInc; //Indica a primeira posicao do vetor 
                              //de incremento das posicoes relevantes
    TNumber threshold;
    
    IGRLVQRD(){
    }
    
    virtual ~IGRLVQRD(){};
        
    //Inicialização com valores aleatorios
    virtual IGRLVQRD& initialize(int nNodes, int ncls, int wsize){

        IGRLVQ::initialize(nNodes, ncls, wsize);
        
        raizVectorRelInc = 0; //Inicializa a raiz apontado para o indice 0
        vectorRelInc.size(wsize);
        vectorRelInc.fill(1);
        vectorMaxRel.size(wsize);
        vectorMaxRel.fill(0);
        threshold = (1.0/wsize)*0.1;
        
        return *this;
    }
    
    virtual TNumber distWeightDerivV(const TVector &w, const TNode &node, int i){
        //Calcula a distancia derivV com pesos
        return 2 * weight[i] * (w[i] - node.w[i]);
        
    }
    
    virtual TNumber distDerivParam(const TVector &w, const TNode &node, int i){
        //Calcula a distancia derivParam
        return (w[i] - node.w[i]) * (w[i] - node.w[i]);
        
    }
    
    virtual TNumber distEuclidianaWeight(const TVector &w, const TNode &node){
        //Calcula a distancia Euclidiana com pesos
        TNumber dist = 0;
        
        for (int i = raizVectorRelInc; i < vectorRelInc.size(); i+=vectorRelInc[i]) {
            dist += weight[i] * (w[i] - node.w[i]) * (w[i] - node.w[i]);

        }         
        return dist;
    }
    
   
    IGRLVQRD& trainning(int N = 1) {
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
                    dbgOut(0) << add->loseCount << "/"<< meshNodeSet.size() << endl;
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
            
            if (i%1000==0){
                //updateVectorRelInc();
                updateVectorRelIncWithMax();
                dbgOut(0) << "Ciclos: " << i << "\tChange: " << total_change << "\tChangeW: " << total_changeW << endl;
            }
            i++;
        
            
        } while (total_changeW > min_change); //total_changeW > min_change;
        
        dbgOut(0) << "Ciclos: " << i << "\tChange: " << total_change << "\tChangeW: " << total_changeW << endl;
        for (int i = 0; i < vectorRelInc.size(); i++) {
            dbgOut(0) << "[" << i << "]: " << vectorRelInc[i];
        }
        dbgOut(0) << endl;

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
        dbgOut(2) << "Acuracia: " << tp/data.rows() << endl;
        return *this;
    }
    
    virtual IGRLVQRD& updateVectorRelInc(){
        TNumber found, increm, j;
        //percorro o vetor de relevancia em busca de relevancia igual a zero
        int i = raizVectorRelInc;
        
        while (i < vectorRelInc.size()) { //Enquanto nao chegar ao final do vetor
            dbgOut(2) << "R[" << i << "]: " << weight[i] << "\t" << "I[" << i << "]: " << vectorRelInc[i] << endl;
            if(weight[i] < threshold){
                found = false;
                increm = vectorRelInc[i]; //guarda o incremento atual
                vectorRelInc[i] = 0; //zera o incremento
                
                while (found == false && i > 0){ //busca o indice mais próximo com valor igual ou maior a 1
                    i--;
                    if(vectorRelInc[i] > 0){
                        vectorRelInc[i] += increm;
                        found = true;
                        i += vectorRelInc[i];
                    } 
                }

                if (found == false) {
                    raizVectorRelInc += increm;
                    i = raizVectorRelInc;
                }
            }
            else{
                i+=vectorRelInc[i];
            }
        }
    }
    
    virtual IGRLVQRD& updateVectorRelIncWithMax(){
        TNumber found, increm, j, min;
        //percorro o vetor de relevancia em busca de relevancia igual a zero
        int i = raizVectorRelInc;
        min = vectorMaxRel.min();
        
        while (i < vectorRelInc.size()) { //Enquanto nao chegar ao final do vetor
            dbgOut(2) << "R[" << i << "]: " << weight[i] << "/" << vectorMaxRel[i] << "\t" << "I[" << i << "]: " << vectorRelInc[i] << endl;
            if(weight[i] < threshold && vectorMaxRel[i] <= min){
                found = false;
                increm = vectorRelInc[i]; //guarda o incremento atual
                vectorRelInc[i] = 0; //zera o incremento
                
                while (found == false && i > 0){ //busca o indice mais próximo com valor igual ou maior a 1
                    i--;
                    if(vectorRelInc[i] > 0){
                        vectorRelInc[i] += increm;
                        found = true;
                        i += vectorRelInc[i];
                    } 
                }

                if (found == false) {
                    raizVectorRelInc += increm;
                    i = raizVectorRelInc;
                }
            }
            else{
                vectorMaxRel[i] = 0;
                i+=vectorRelInc[i];
            }
        }
    }
    
    virtual IGRLVQRD& trainningStep() {
        
        for(int i=0; i < data.rows(); i++){
            TVector v;
            int row = rand()%data.rows();
            data.getRow(row, v);
            updateMap(v, vcls[row]);        
        }

        return *this;
    }
    
    virtual IGRLVQRD& trainningEach(int N = 1) {
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
    
    virtual IGRLVQRD& updateMap(const TVector &w, int cls) {
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
                        
                for (int i = raizVectorRelInc; i < vectorRelInc.size(); i+=vectorRelInc[i]) {
                    TNumber updateW, updateR;
                    updateW = factorW * (distDeriv(w, *winner, i));
                    updateR = factorR * (distDeriv(w, *runnerUp, i));

                    //Atualiza os prototipos Winner e Runnerup
                    if(updateW && updateR == 0) continue;
                    winner->w[i] = winner->w[i] + updateW;
                    runnerUp->w[i] = runnerUp->w[i] + updateR;

                }

                updateDistanceMeasure(w, *winner, *runnerUp);

                //Calcula o erro de classificacao
                if (dW > dR) { //O Nodo Runner esta mais proximo da amostra que o Winner
                    //Guarda a amostra com a menor distância para cada classe
                    runnerUp->loseCount++;
                }
            }
        }

        learningDecay(); // Faz o decaimento dos valores dos parâmetros
    }
    
    virtual IGRLVQRD& updateDistanceMeasure(const TVector &w, const TNode &winner, const TNode &runnerUp){
        
        TNumber dR, dW, sqsum, xW, xR, sgd;
        TVector deltaW(w.size());
        
        dR = distWeight(w, runnerUp);
        dW = distWeight(w, winner);
        sqsum = (dW + dR) * (dW + dR);
        sgd = sgdprime((dW - dR) / (dW + dR)); //Calcula sigmoide
        xW = 2 * dR / sqsum;
        xR = 2 * dW / sqsum;

        //Calculo da relevancia com problemas????
        for (int i = raizVectorRelInc; i < vectorRelInc.size(); i+=vectorRelInc[i]) {      
            deltaW[i] = - alpha_w * sgd * ((xW * distDerivP(w, winner, i)) - (xR * distDerivP(w, runnerUp, i)));
        }
        updateWeightVector(deltaW);
    }
    
    virtual IGRLVQRD& updateWeightVector(const TVector deltaW){
        TNumber weight_change = 0;
        TNumber sum = 0;
        TNumber min, max;
                  
        for (int i = raizVectorRelInc; i < vectorRelInc.size(); i+=vectorRelInc[i]) {
            weight[i] += deltaW[i];
            weight_change += deltaW[i] * deltaW[i];
            sum += weight[i];
         }
        total_changeW += sqrt(weight_change);
        
        //Normaliza o vetor para um intervalo [0,1] se existir valor negativo
        weight.minMax(min, max);
        if (min < 0){
            sum = 0;
            for (int j = raizVectorRelInc; j < vectorRelInc.size(); j+=vectorRelInc[j]) {
                weight[j] = (weight[j] - min)/(max-min);
                dbgOut(2) << "[" << j << "]:" << weight[j] << "\t";
                sum += weight[j];
            }
            dbgOut(2) << endl;
        }
        
        for (int i = raizVectorRelInc; i < vectorRelInc.size(); i+=vectorRelInc[i]) {
            weight[i] = weight[i] * (1/sum);
            
            if(weight[i] > vectorMaxRel[i]){
                vectorMaxRel[i] = weight[i];
            } 
        }
    }
    
};

#endif	/* IGRLVQRD_H */

