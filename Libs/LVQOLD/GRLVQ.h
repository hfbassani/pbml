/* 
 * File:   GRLVQ.h
 * Author: flavia
 *
 * Created on 21 de Maio de 2014, 11:08
 */

#ifndef GRLVQ_H
#define	GRLVQ_H

#include "LVQ.h"
#include <map>
#include "DebugOut.h"
#include "NodeW.h"
#include "LVQ21.h"
#include "GLVQ.h"


#define distWeight(x,y)   distEuclidianaWeight((x),(y))
#define distDeriv(x,y,i)  distWeightDerivV((x),(y),(i))
#define distDerivP(x,y,i) distDerivParam((x),(y),(i))

class GRLVQ: public LVQ<LVQNode> {

public:
    TNumber alpha_tp;    //Learn positive 
    TNumber alpha_tp0;   //Initial learn positive
    TNumber alpha_tn;    //Learn negative 
    TNumber alpha_tn0;   //Initial learn negative
    TNumber alpha_w;     //Learning weight
    TNumber alpha_w0;    //Initial learning weight
    TNumber tau;         //decay rate
    TNumber t;           //epochs
    TNumber tmax;        //epochs max
    TVector weight;      //weight vector
    
    TVector maxWeight;   //limit for the weights
    TVector minWeight;   //limit for the weights
    
    TNumber total_change;
    TNumber total_changeW;
    TNumber min_change; //minimal chanve values for prototype adjustment
    
   
    GRLVQ(){
    }
    
    virtual ~GRLVQ(){};
    
    //Hans: set max weights
    void setMaxWeights(TVector &maxWeight) {
        this->maxWeight = maxWeight;
    }

    //Hans: set max weights
    void setMinWeights(TVector &minWeight) {
        this->minWeight = minWeight;
    }
        
    //Random inicitialization
    virtual GRLVQ& initialize(int nNodes, int ncls, int wsize){

        alpha_tp = alpha_tp0;
        alpha_tn = alpha_tn0;
        alpha_w = alpha_w0;
        min_change = min_change; //Finalize the training 
        meshNodeSet.clear();
        
        t = 0; //epochs count

        int cls = 0;        

        //Create a new node with random values
        for (int x = 0; x < nNodes; x++){
            TVector v(wsize); 
            v.random().mult(0.1).add(0.95); 
            LVQNode* node = new LVQNode(x, v, cls);
            aloc_node++;
            meshNodeSet.insert(node);
            
            node->winCount = 0; //inicializa o nodo como nunca ter sido vencedor (usado para remocao dos nodos)
            node->loseCount = 0;
            node->totalchange = 0;
//            node->ciclesCount = 0;
            
            cls++;
            
            if (cls == ncls) {
                cls = 0;
            }
        }
        
        //Inicializa o vetor de pesos com uma distribuicao uniforme
        weight.size(wsize);
        weight.fill(1.0/wsize);
        
        //Hans: initialize max and min weights
//        if (maxWeight.size()!=wsize) {
//            maxWeight.size(wsize);
//            maxWeight.fill(1);
//        }
//        if (minWeight.size()!=wsize) {
//            minWeight.size(wsize);
//            minWeight.fill(0);
//        }
        
        dbgOut(2) << "Relevances: " << weight.toCSV() << endl;
        
        return *this;
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
        
        for (int i = 0; i < node.w.size(); i++) {
            dist += weight[i] * (w[i] - node.w[i]) * (w[i] - node.w[i]);

        }         
        return dist;
    }
    
    //Distancias modificadas para que os dados 0, 1 e 2 tenham a mesma distancia 1
    
    virtual TNumber limitDist(TNumber w, TNumber n){
        
        TNumber d = w - n;
        
        if (d < -1){
            return -1;
        }
        else{
            if(d > 1){
                return 1;
            }
        }
        return d;
        
    }
    
    virtual TNumber distPseudoWeightDerivV(const TVector &w, const TNode &node, int i){
        //Calcula a distancia derivV com pesos
        TNumber di = limitDist(w[i], node.w[i]);
        return 2 * weight[i] * di;
        
    }
    
    virtual TNumber distPseudoDerivParam(const TVector &w, const TNode &node, int i){
        //Calcula a distancia derivParam
        TNumber di = limitDist(w[i], node.w[i]);
        return di*di;
        
    }
    
    virtual TNumber distPseudoEuclidianaWeight(const TVector &w, const TNode &node){
        //Calcula a dissimilaridade com pesos
        TNumber dist = 0;
        
        for (int i = 0; i < node.w.size(); i++) {
            TNumber di = limitDist(w[i], node.w[i]);
            dist += weight[i] * di * di;

        }         
        return dist;
    }
    
    
    //Calculo das distancias Quadraticas
    //Artigo Hammer, 2005. Supervised Neural Gas with General Similarity Measure
    
    virtual TNumber distWeightDerivVQuartic(const TVector &w, const TNode &node, int i){
        //Calcula a distancia derivV Quadratica com pesos
        return 4 * (weight[i] * weight[i]) * ((w[i] - node.w[i]) * (w[i] - node.w[i]) * (w[i] - node.w[i]));
        
    }
    
    virtual TNumber distDerivParamQuartic(const TVector &w, const TNode &node, int i){
        //Calcula a distancia derivParam Quadratica
        return 2 * weight[i] * ((w[i] - node.w[i]) * (w[i] - node.w[i]) * (w[i] - node.w[i]) * (w[i] - node.w[i]));
        
    }
     
    virtual TNumber distEuclidianaWeightQuartic(const TVector &w, const TNode &node){
        //Calcula a distancia Euclidiana Quadratica com pesos
        TNumber dist = 0;
        
        for (int i = 0; i < node.w.size(); i++) {
            dist += (weight[i] * weight[i]) * ((w[i] - node.w[i]) * (w[i] - node.w[i]) * (w[i] - node.w[i]) * (w[i] - node.w[i]));
        }
        
        return dist;
    }

    
    virtual GRLVQ& getWinnerRunnerUpWeight(const TVector &w, TNode* &winner, TNode* &runnerUp, int cls) {
        TNumber dw = std::numeric_limits<TNumber>::max();
        TNumber dr = std::numeric_limits<TNumber>::max();
        
        winner = NULL;
        runnerUp = NULL;
        
        TPNodeSet::iterator it;
        for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end(); it++) {
            TNode* node = *it;
            if ((*it)->cls == cls) {
                TNumber tempW = distWeight(w, *node);
                if (dw > tempW) {
                    dw = tempW;
                    (*it)->winCount = (*it)->winCount + 1; //adiciona +1 ao contador de vitorias
                    winner = (*it);
                }
            }
            else {
                TNumber tempR = distWeight(w, *node);
                if (dr > tempR) {
                    dr = tempR;
                    runnerUp = (*it);
                }
            }
        }

        return *this;
    }
    GRLVQ& trainning(int N = 1) {
        int i = 0;
       
        do{
            total_change = 0;
            total_changeW = 0;
            trainningStep();
            i++;
        } while (i < N && total_change > min_change);     
        return *this;
    }
    
    virtual GRLVQ& trainningStep() {

        for(int i=0; i < data.rows(); i++){
            TVector v;
            if (data.rows()>0) {
                int row = rand()%data.rows();
                data.getRow(row, v);
                updateMap(v, vcls[row]);        
            }
        }
        return *this;
    }
    
    virtual GRLVQ& trainningEach(int N = 1) {
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
            dbgOut(2) << "It: " << n << "\t" << "Change Proto: " << total_change << endl;
            
        }
        
        return *this;
    }

    virtual GRLVQ& updateMap(const TVector &w, int cls) {
        TNode *winner, *runnerUp;
                
        //Get the closest prototypes with differents classes
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
                total_change += (sqrt(proto_changeRunner) + sqrt(proto_changeWinner));
            }
        }
        learningDecay();
    }
    
    virtual GRLVQ& updateDistanceMeasure(const TVector &w, const TNode &winner, const TNode &runnerUp){
        
        TNumber dR, dW, sqsum, xW, xR, sgd;
        TVector deltaW(w.size());
        
        dR = distWeight(w, runnerUp);
        dW = distWeight(w, winner);
        sqsum = (dW + dR) * (dW + dR);
        sgd = sgdprime((dW - dR) / (dW + dR)); //Calcula sigmoide
        xW = 2 * dR / sqsum;
        xR = 2 * dW / sqsum;

        //Calculo da relevancia com problemas????
        for (int i = 0; i < deltaW.size(); i++) {      
            deltaW[i] = - alpha_w * sgd * ((xW * distDerivP(w, winner, i)) - (xR * distDerivP(w, runnerUp, i)));
        }
        updateWeightVector(deltaW);
    }
    
    virtual GRLVQ& updateWeightVector(const TVector deltaW){
        TNumber weight_change = 0;
                  
        for (int i = 0; i < deltaW.size(); i++) {
            weight[i] += deltaW[i];
//            if(isnan(weight[i])){
//                    dbgOut(0) << weight.toString() << endl;
//                    dbgOut(0) << deltaW.toString() << endl;
//            }
            weight_change += deltaW[i] * deltaW[i];
            
            //Hans: Limit weights
//            if (weight[i]>maxWeight[i])
//                weight[i] = maxWeight[i];
//            if (weight[i]<minWeight[i])
//                weight[i] = minWeight[i];
//            if(isnan(weight[i])){
//                dbgOut(2) << weight[i] << " ,i:" << i << endl;  
//            }
        }
        total_changeW += sqrt(weight_change);
                
            
        //Normaliza, caso tenha valores menores que zero no vetor de pesos
        /*
        TNumber min;
        min = weight.min();
        if (min < 0) {
            for (int j = 0; j < weight.size(); j++) {
                weight[j] -= min;
            }
        }
        */
        
        //*Normaliza, atribuindo zero para relevancias menores que zero
        //http://matlabserver.cs.rug.nl/gmlvqweb/web/
        for (int j = 0; j < weight.size(); j++) {
            if (weight[j] < 0){ //(1.0/weight.size())/3.0
                weight[j] = 0; 
            }
            //dbgOut(0) << weight[0] << weight[1] << weight[2] << endl;
//            total_changeW += sqrt(weight[j]*weight[j]);
//            if(isnan(weight[j])){
//                dbgOut(2) << weight[j] << " ,j:" << j << endl;  
//            }
        }
        
        

        //*/
        //Normaliza os valores pela soma do vetor = 1. 
//        dbgOut(0) << "Multiplica" << endl;

//        if(weight.sum()==0){
//            dbgOut(0) << weight.toString() << endl; 
//            dbgOut(0) << deltaW.toString() << endl;
//        }
        
        weight.mult(1/weight.sum());
//        for (int j = 0; j < weight.size(); j++) {
//            total_changeW += weight[j]/data.rows();
//        }
        
//        for (int j = 0; j < weight.size(); j++) {
//            if(isnan(weight[j])){
//                dbgOut(0) << weight[j] << " ,j:" << j << endl;  
//            }
//        }
//        dbgOut(0) << endl;
        
    }
    
    virtual GRLVQ& learningDecay() {
        
        t++;
        
        alpha_tp = (alpha_tp0)/(1 + tau * (t - 0)); //Equacao (Nova e Estevez, 2013 Neural Comput&Applic)
        if (alpha_tp <= 0) alpha_tp = 0.000001;
        alpha_tn = (alpha_tn0)/(1 + tau * (t - 0));
        if (alpha_tn <= 0) alpha_tn = 0.000001;
        alpha_w = (alpha_w0)/(1 + tau * (t - 0));
        if (alpha_w <= 0) alpha_w = 0.000001;
        
        
    }
};

#endif	/* GRLVQ_H */

