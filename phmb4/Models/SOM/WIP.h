/*
 * SSSOM.h
 *
 *  Created on: 2017
 *      Author: phmb4
 */

#ifndef SSSOM_H_
#define SSSOM_H_

#include <set>
#include <map>
#include <vector>
#include "Mesh.h"
#include "MatUtils.h"
#include "MatVector.h"
#include "MatMatrix.h"
#include "SOM.h"
#include "SSSOMNode.h"
#include <chrono>
#include <random>

#define qrt(x) ((x)*(x))

using namespace std;


class WIP : public SOM<WIPNode> {
public:
    uint maxNodeNumber;
    int epochs;
    float minwd;
    float e_b;
    float e_n;
    float e_var;
    
    TNumber dsbeta; //Taxa de aprendizagem
    TNumber epsilon_ds; //Taxa de aprendizagem
    float age_wins;       //period to remove nodes
    float lp;           //remove percentage threshold

    int nodeID;
    
    int unsup_win;
    int unsup_create;
    int unsup_else;
    int sup_win;
    int sup_create;
    int sup_else;
    int sup_handle_new_win_full;
    int sup_handle_new_win_relevances;
    int sup_handle_create;
    int sup_handle_else;
    
    inline float activation(TNode *node, const TVector &w) {
        float distance = 0;
        node->region = true;

        for (uint i = 0; i < w.size(); i++) {
            float diff = qrt((w[i] - node->w[i]));
            distance += node->ds[i] * diff;
            
            float var = node->a_corrected[i] / node->ds[i];
            if (w[i] <= node->w[i] - var || w[i] >= node->w[i] + var)
                node->region = false;
            //node->region += (diff / (qrt(var) + 0.0000001));
        }
        
        float sum = node->ds.sum();
        
//        float f_distance = (distance) / (sum + 0.0000001);
        
//        return 1 - f_distance;
        
//        return sqrt(distance);
        return (sum / (sum + (distance) + 0.0000001));

    }
    
    inline float wdist(const TNode &node1, const TNode &node2) {
        float distance = 0;

        for (uint i = 0; i < node1.ds.size(); i++) {
            distance +=  qrt((node1.ds[i] - node2.ds[i]));
        }
        
        return sqrt(distance);
    }
    
    void updateRelevances(TNode &node, const TVector &w, TNumber e){
        node.count += 1;
        
        float beta = dsbeta;
        //update averages
        for (uint i = 0; i < node.a.size(); i++) {
            //update neuron weights
            float distance = fabs(w[i] - node.w[i]);
            node.a[i] = beta * node.a[i] + (1 - beta) * distance;
            node.a_corrected[i] = node.a[i] / (1 - pow(beta, node.count));
        }

        float max = node.a_corrected.max();
        float min = node.a_corrected.min();
        float average = node.a_corrected.mean();
        
        //update neuron ds weights
        for (uint i = 0; i < node.a_corrected.size(); i++) {
            if ((max - min) != 0) {
                //node.ds[i] = 1 - (node.a[i] - min) / (max - min);
                node.ds[i] = 1/(1+exp((node.a_corrected[i]-average)/((max - min)*epsilon_ds)));
            }
            else
                node.ds[i] = 1;
        }
    }

    inline void updateNode(TNode &node, const TVector &w, TNumber e) {
        
        updateRelevances(node, w, e);
        
        //Passo 6.1: Atualiza o peso do vencedor
        //Atualiza o nó vencedor
        node.w = node.w + e * (w - node.w);      
    }

    WIP& updateConnections(TNode *node) {
        
        TPNodeSet::iterator itMesh = meshNodeSet.begin();
            
        while (itMesh != meshNodeSet.end()) {
            if (*itMesh != node) {
                if (checkConnectivity(node, *itMesh)) {
                    if (!isConnected(node, *itMesh))
                    connect(node, *itMesh);
                } else {
                    if (isConnected(node, *itMesh))
                        disconnect(node, *itMesh);
                }
            }
            itMesh++;
        }
        return *this;
    }
    
    bool checkConnectivity(TNode *node1, TNode *node2) {
        
        if((node1->cls == noCls || node2->cls == noCls || node1->cls == node2->cls) && wdist(*node1, *node2)<minwd) {
            return true;
        }
        
        return false;
        
    }
    
    WIP& updateAllConnections() {

        //Conecta todos os nodos semelhantes
        TPNodeSet::iterator itMesh1 = meshNodeSet.begin();
        while (itMesh1 != meshNodeSet.end()) {
            TPNodeSet::iterator itMesh2 = meshNodeSet.begin();
            
            while (itMesh2 != meshNodeSet.end()) {
                if (*itMesh1!= *itMesh2) {
                    if (checkConnectivity(*itMesh1, *itMesh2)) {
                        if (!isConnected(*itMesh1, *itMesh2))
                        connect(*itMesh1, *itMesh2);
                    } else {
                        if (isConnected(*itMesh1, *itMesh2))
                            disconnect(*itMesh1, *itMesh2);
                    }
                }
                itMesh2++;
            }
            
            itMesh1++;
        }

        return *this;
    }

    WIP& removeLoosers() {

//        enumerateNodes();

        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {
            if (meshNodeSet.size()<2)
                break;

            if ((*itMesh)->wins < step*lp) {
                eraseNode((*itMesh));
                itMesh = meshNodeSet.begin();
                
            } else {
                itMesh++;
                
            }
        }
        
        return *this;
    }
    
    WIP& trainningStep(int row,  std::vector<int> groups, std::map<int, int> &groupLabels) {
        TVector v(data.cols());
        for (uint l = 0; l < data.cols(); l++)
                v[l] = data[row][l];
        
        chooseTrainingType(v, groups[row]);
        
        return *this;
    }
    
    void runTrainingStep(bool sorted, std::vector<int> groups, std::map<int, int> &groupLabels) {
        if (sorted) {
            trainningStep(step%data.rows(), groups, groupLabels);
        } else {
            trainningStep(rand()%data.rows(), groups, groupLabels);
        }
    }
    
    void chooseTrainingType(TVector &v, int cls) {
        
        if (cls != noCls) { 
            updateMapSup(v, cls);
        } else { 
            updateMap(v); 
        }
    }
    
    WIP& finishMapFixed(bool sorted, std::vector<int> groups, std::map<int, int> &groupLabels) {

        dbgOut(2) << "Finishing map with: " << meshNodeSet.size() << endl;
        while (step!=1) { // finish the previous iteration
            runTrainingStep(sorted, groups, groupLabels);
        }
        maxNodeNumber = meshNodeSet.size(); //fix mesh max size
        
        dbgOut(2) << "Finishing map with: " << meshNodeSet.size() << endl;
        
        //step equal to 2
        runTrainingStep(sorted, groups, groupLabels);
        
        while (step!=1) {
            runTrainingStep(sorted, groups, groupLabels);
        }
        
        dbgOut(2) << "Finishing map with: " << meshNodeSet.size() << endl;
        
        return *this;
    }

    WIP& resetWins() {

        //Remove os perdedores
        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {
             (*itMesh)->wins = 0;
             itMesh++;
        }

        return *this;
    }

    TNode* createNodeMap (const TVector& w, int cls) {
        // cria um novo nodo na posição da amostra
        TVector wNew(w);
        TNode *nodeNew = createNode(nodeID, wNew);
        nodeID++;
        nodeNew->cls = cls;
        nodeNew->wins = step * lp;
        
        updateConnections(nodeNew);
        
        return nodeNew;
    }
    
    void ageWinsCriterion(){
        
        //Passo 9:Se atingiu age_wins
        if (step >= age_wins) {
            
            removeLoosers();
            resetWins();
            updateAllConnections();
            
            step = 0;
        }
    }
    
    WIP& updateMap(const TVector &w) {

        using namespace std;
        
        if (meshNodeSet.empty()) {
            createNodeMap(w, noCls);
            
        } else {
            TNode *winner1 = 0;

            winner1 = getFirstWinner(w); //winner
            
            //Se a ativação obtida pelo primeiro vencedor for menor que o limiar
            //e o limite de nodos não tiver sido atingido    
            //bool belongsToWinner = winner1->represents(w);
            if (!winner1->region && (meshNodeSet.size() < maxNodeNumber)) {
                updateRelevances(*winner1, w, e_var);
                createNodeMap(w, noCls);
                
                unsup_create++;
                
            } else if (winner1->region) { // caso contrário
                
                winner1->wins++;
                // Atualiza o peso do vencedor
                updateNode(*winner1, w, e_b);

                //Passo 6.2: Atualiza o peso dos vizinhos
                TPNodeConnectionMap::iterator it;
                for (it = winner1->nodeMap.begin(); it != winner1->nodeMap.end(); it++) {            
                    TNode* node = it->first;
                    updateNode(*node, w, e_n);
                }
                
                unsup_win++;
            } else {
                unsup_else++;
                updateRelevances(*winner1, w, e_var);
            }
        }

        ageWinsCriterion();
        
        step++;
        
        return *this;
    }
    
    WIP& updateMapSup(const TVector& w, int cls) {
        using namespace std;
        
        if (meshNodeSet.empty()) { // mapa vazio, primeira amostra
            createNodeMap(w, cls);
            
        } else {
            TNode *winner1 = 0;
            winner1 = getFirstWinner(w); // encontra o nó vencedor
  
            if (winner1->cls == cls || winner1->cls == noCls) { // winner1 representativo e da mesma classe da amostra
                if (!winner1->region && (meshNodeSet.size() < maxNodeNumber)) {
                    // cria um novo nodo na posição da amostra
                    updateRelevances(*winner1, w, e_var);
                    createNodeMap(w, cls);
                    
                    sup_create++;
                    
                } else if (winner1->region){
                    winner1->wins++;
                    winner1->cls = cls;
                    updateNode(*winner1, w, e_b);
                    updateConnections(winner1);
                    
                    TPNodeConnectionMap::iterator it;
                    for (it = winner1->nodeMap.begin(); it != winner1->nodeMap.end(); it++) {            
                        TNode* node = it->first;
                        updateNode(*node, w, e_n);
                    }
                    
                    sup_win++;
                } else {
                    sup_else++;
                    updateRelevances(*winner1, w, e_var);
                    // winner1->wins++;
                    // updateRelevances(*winner1, w, e_b);
                    
                    // TPNodeConnectionMap::iterator it;
                    // for (it = winner1->nodeMap.begin(); it != winner1->nodeMap.end(); it++) {            
                    //     TNode* node = it->first;
                    //     updateRelevances(*node, w, e_n);
                    // }
                }
                
            } else { // winner tem classe diferente da amostra
                // caso winner seja de classe diferente, checar se existe algum
                // outro nodo no mapa que esteja no raio a_t da nova amostra e
                // que pertença a mesma classe da mesma

                handleDifferentClass(winner1, w, cls);
            }   
        }

        ageWinsCriterion();
        
        step++;
        
        return *this;
    }
    
    void handleDifferentClass(TNode *winner1, const TVector& w, int cls) {
        TNode *newWinner = winner1;
        while((newWinner = getNextWinner(newWinner)) != NULL) { // saiu do raio da ativação -> não há um novo vencedor
            if (newWinner->cls == cls || newWinner->cls == noCls) { // novo vencedor valido encontrado
                break;
            }
        }

        if (newWinner != NULL) { // novo winner de acordo com o raio de a_t
            
            // empurrar o primeiro winner que tem classe diferente da amostra
//            updateNode(*winner1, w, -e_n);
            
            newWinner->wins++;
            
            if (newWinner->region) {
                // puxar o novo vencedor
                updateNode(*newWinner, w, e_b);
                
//                if (newWinner->region) {
//                    newWinner->cls = cls;
//                    updateConnections(newWinner);
//                }
                TPNodeConnectionMap::iterator it;
                for (it = newWinner->nodeMap.begin(); it != newWinner->nodeMap.end(); it++) {            
                    TNode* node = it->first;
                    updateNode(*node, w, e_n);
                    // updateRelevances(*node, w, e_n);
                }
                
                sup_handle_new_win_full++;
           } else {
               updateRelevances(*newWinner, w, e_var);
               sup_handle_new_win_relevances++;
           }
           
        } else if (meshNodeSet.size() < maxNodeNumber) {
            
            // cria um novo nodo na posição da amostra
//            updateNode(*winner1, w, -e_n);
//            updateRelevances(*winner1, w, e_var);
//            createNodeMap(w, cls);
            
            sup_handle_create++;
            TVector wNew(winner1->w);
            TNode *nodeNew = createNodeMap(wNew, cls);
            
            TVector aNew(winner1->a);
            nodeNew->a = aNew;
            
            
            TVector aCorrNew(winner1->a_corrected);
            nodeNew->a_corrected = aCorrNew;
            
            TVector dsNew(winner1->ds);
            nodeNew->ds = dsNew;   
            
            updateNode(*nodeNew, w, e_b);

            updateNode(*winner1, w, -e_n);
            
        } else if (newWinner == NULL) {
            updateRelevances(*winner1, w, e_var);
            
            // TPNodeConnectionMap::iterator it;
            // for (it = newWinner->nodeMap.begin(); it != newWinner->nodeMap.end(); it++) {            
            //     TNode* node = it->first;
            //     updateRelevances(*node, w, e_n);
            // }
//            updateNode(*winner1, w, -e_n);
            sup_handle_else++;
        }
    }

    virtual TNode *getFirstWinner(const TVector &w){
        TNode *winner = 0;
        
        winner = (*Mesh<TNode>::meshNodeSet.begin());
        winner->act = activation(winner, w);

        TNumber act = winner->act;
        
        TPNodeSet::iterator it;
        it = Mesh<TNode>::meshNodeSet.begin();
        it++;
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {
            (*it)->act = activation((*it), w);
            if ((*it)->act > act) {
                act = (*it)->act;
                winner = (*it);
            }
        }

        return winner;
    }

    virtual TNode *getNextWinner(TNode *previowsWinner) {
        previowsWinner->act = 0;
        
        TNode *winner = 0;
        winner = (*Mesh<TNode>::meshNodeSet.begin());
        TNumber winnerAct = winner->act;

        TPNodeSet::iterator it;
        it = Mesh<TNode>::meshNodeSet.begin();
        it++;
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {

            if ((*it)->act > winnerAct) {
                winnerAct = (*it)->act;
                winner = (*it);
            }
        }
        
        if (winner->act == 0)
            return NULL;

        return winner;
    }
    
    virtual inline std::vector<int> getWinnerResult(const TVector &w) {
        TNode *winner = getFirstWinner(w);
        
        if (winner->cls == noCls) {
            TNode *newWinner = winner;
            while((newWinner = getNextWinner(newWinner)) != NULL) { // saiu do raio da ativação -> não há um novo vencedor
                if (newWinner->cls != noCls) { // novo vencedor valido encontrado
                    break;
                }
            }

            if (newWinner != NULL && newWinner->cls != noCls) 
                winner = newWinner;
        }
        
        std::vector<int> result;
        result.push_back(getNodeIndex(*winner));
        result.push_back(winner->cls);
        result.push_back(winner->getId());
        result.push_back(winner->act);
        
        return result;
    }
    
    int getNodeIndex(WIPNode &node) {
        SOM<WIPNode>::TPNodeSet::iterator it = meshNodeSet.begin();
        int i = 0;
        for (; it != meshNodeSet.end(); it++, i++) {
            if ((*it) == &node) {
                return i;
            }
        }
        return -1;
    }
    
    bool isNoise(const TVector &w) {
        TNode *winner = getFirstWinner(w);
        return winner->region;
    }

    void reset(int dimw) {
        WIP::dimw = dimw;
        step = 0;

        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;
        nodeID = 0;

        destroyMesh();
    }

    void reset(void) {
        reset(dimw);
    }

    WIP(int dimw) {
        
    };

    ~WIP() {
    }
};

#endif /* SSSOM_H_ */
