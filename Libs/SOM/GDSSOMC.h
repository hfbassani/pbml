/*
 * GDSSOMC.h
 *
 *  Created on: 10/02/2009
 *      Author: daniel
 */

#ifndef GDSSOMC_H_
#define GDSSOMC_H_

#include <set>
#include <map>
#include <vector>
#include "Mesh.h"
#include "MatUtils.h"
#include "MatVector.h"
#include "MatMatrix.h"
#include "SOM.h"
#include "DSNode.h"
#include "GDSSOM.h"

#define qrt(x) ((x)*(x))

using namespace std;

class GDSSOMC : public SOM<GDSNode> {
public:
    uint maxNodeNumber;
    float minwd;
    float e_b;
    float e_n;
    int nodesCounter;
    
    TNumber dsbeta; //Taxa de aprendizagem
    TNumber epsilon_ds; //Taxa de aprendizagem
    int age_wins;       //period to remove nodes
    float lp;           //remove percentage threshold
    float a_t;
    
    int nodesLeft;

    inline float activation(const TNode &node, const TVector &w) {

        float distance = 0;

        for (uint i = 0; i < w.size(); i++) {
            distance += node.ds[i] * qrt((w[i] - node.w[i]));
        }

        TNumber norm = node.ds.norm();
        return (norm / (norm + distance*distance + 0.0000001));
    }

    inline float dist2(const TNode &node, const TVector &w) {
        /*float distance = 0;

        for (uint i = 0; i < w.size(); i++) {
            distance += node.ds[i] * qrt((w[i] - node.w[i]));
        }

        return distance / (node.ds.sum() + 0.0000001);*/
        return 1/activation(node, w);
    }

    inline float dist(const TNode &node, const TVector &w) {
        return sqrt(dist2(node, w));
    }

    inline float wdist(const TNode &node1, const TNode &node2) {
        float distance = 0;

        for (uint i = 0; i < node1.ds.size(); i++) {
            distance +=  qrt((node1.ds[i] - node2.ds[i]));
        }
        
        return sqrt(distance);
    }

    inline void updateNode(TNode &node, const TVector &w, TNumber e) {
        
        //update averages
        for (uint i = 0; i < node.a.size(); i++) {
            //update neuron weights
            float distance = fabs(w[i] - node.w[i]);
            node.a[i] = e*dsbeta* distance + (1 - e*dsbeta) * node.a[i];
        }

        float max = node.a.max();
        float dsa = node.ds.mean();

        //update neuron ds weights
        for (uint i = 0; i < node.a.size(); i++) {
            if (max != 0)
                node.ds[i] = 1 - (node.a[i] / max);
            else
                node.ds[i] = 1;

            if (node.ds[i] < 0.95*dsa)
                node.ds[i] = epsilon_ds;
            
            if (node.ds[i] < epsilon_ds)
                node.ds[i] = epsilon_ds;
        }

        //Passo 6.1: Atualiza o peso do vencedor
        //Atualiza o nó vencedor
        node.w = node.w + e * (w - node.w);
    }
    
    GDSSOMC& removeLoosers() {

        //Remove os perdedores
//        dbgOut(1) << "\nSize:\t" << meshNodeSet.size() << endl;
//        dbgOut(1) << "NodesLeft:\t" << nodesLeft << endl;
//        dbgOut(1) << "step:\t" << step << endl;
//        dbgOut(1) << "winsPerNode:\t" << winsPerNode << endl;
        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {
            if (meshNodeSet.size()<2)
                break;

            if ((*itMesh)->wins < step*lp) {
                //dbgOut(1) << (*itMesh)->wins << "\t<\t" << winsPerNode*lp << endl;
                eraseNode((*itMesh));
                itMesh = meshNodeSet.begin();
            } else {
                //dbgOut(1) << (*itMesh)->wins << "\t>=\t" << winsPerNode*lp << endl;
                itMesh++;
            }
        }


        return *this;
    }
    
    GDSSOMC& resetWins() {

        //Remove os perdedores
        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {
             (*itMesh)->wins = 0;
             itMesh++;
        }

        return *this;
    }

    GDSSOMC& updateMap(const TVector &w) {

        using namespace std;

        TNode *winner1 = NULL;
        TNode *winner2 = NULL;

        //Passo 3 : encontra os dois nós vencedores
        getWinners(w, winner1, winner2); //winner
        winner1->wins++;

        if (winner2!=NULL)
        if (!isConnected(winner1, winner2)) {
            //Se os nós não estão conectado cria uma conexão entre os nós com age igual a zero
            connect(winner1, winner2);
        }

        //Passo 6: Calcula a atividade do nó vencedor
        TNumber a = activation(*winner1, w); //DS activation
        
        //Se a ativação obtida pelo primeiro vencedor for menor que o limiar
        //e o limite de nodos não tiver sido atingido

        if ((a < a_t) && (meshNodeSet.size() < maxNodeNumber)) {
            //dbgOut(2) << a << "\t<\t" << a_t << endl;
            //Cria um novo nodo no local do padrão observado
            TVector wNew(w);
            TNode *nodeNew = createNode(0, wNew);
            nodeNew->wins = step/meshNodeSet.size();
            
            /*
            connect(winner1, nodeNew);
            if (winner2!=NULL)
                connect(winner2, nodeNew);

            //disconnect(winner1, winner2);
            TNumber d12 = winner1->w.dist(winner2->w);
            TNumber d1New = winner1->w.dist(w);
            TNumber d2New = winner2->w.dist(w);

            if ((d12 < d1New) && (d12 < d2New)) {
                //Não disconecta wiiner1 de winner2, pois já estão com a menor distância
                if (d1New < d2New)//Conecta a segunda menor distância
                {
                    connect(winner1, nodeNew);
                } else {
                    connect(winner2, nodeNew);
                }
            } else if ((d1New < d12) && (d1New < d2New)) {
                    
                connect(winner1, nodeNew);
                if (d2New < d12) {//Segunda menor distância
                    disconnect(winner1, winner2);
                    connect(winner2, nodeNew);
                }//Se não não faz nada pois winner1 e winner2 já estão conectados

            } else
                if ((d2New < d12) && (d2New < d1New)) {
                connect(winner2, nodeNew);
                if (d1New < d12) {
                    disconnect(winner1, winner2);
                    connect(winner1, nodeNew);
                }//Se não faz nada pois winner1 e winner2 já estão conectados
            }
            */

        } else { // caso contrário
            // Atualiza o peso do vencedor
            updateNode(*winner1, w, e_b);

            //Passo 6.2: Atualiza os vizinhos
            TPNodeConnectionMap::iterator it;
            for (it = winner1->nodeMap.begin(); it != winner1->nodeMap.end(); it++) {            
                TNode* node = it->first;
                updateNode(*node, w, e_n);
                updateNodeConnections(*node);
            }
        }

        //Passo 9:Se atingiu age_wins
        if (step >= age_wins) {

            int size = meshNodeSet.size();
            //remove os perdedores
            removeLoosers();
            dbgOut(1) << size << "\t->\t" << meshNodeSet.size() << endl;
            //reseta o número de vitórias
            resetWins();
            step = 0;
        }

        step++;
        return *this;
    }

    inline void updateNodeConnections(TNode &node) {

        //Disconnect different neighbors
        TPNodeConnectionMap::iterator it;
        
        if (node.nodeMap.size()<2)
            return;
        
        for (it = node.nodeMap.begin(); it != node.nodeMap.end(); it++) {
            TNode *neighbor = (*it).first;
            if (wdist(node, *neighbor) > minwd) {
                
                if (neighbor->nodeMap.size()>1)
                        disconnect(&node, neighbor);
            }
        }
    }
    
    virtual TNode *getFirstWinner(const TVector &w){
        return getWinner(w);
    }

    inline TNode* getWinner(const TVector &w) {
        TNode *winner = 0;
        TNumber temp = 0;
        TNumber d = dist(*(*Mesh<TNode>::meshNodeSet.begin()), w);
        winner = (*Mesh<TNode>::meshNodeSet.begin());

        TPNodeSet::iterator it;
        it = Mesh<TNode>::meshNodeSet.begin();
        it++;
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {
            temp = dist(*(*it), w);
            if (d > temp) {
                d = temp;
                winner = (*it);
            }
        }

        return winner;
    }

    inline GDSSOMC& getWinners(const TVector &w, TNode* &winner1, TNode* &winner2) {
        TPNodeSet::iterator it = Mesh<TNode>::meshNodeSet.begin();
        TNumber minDist = dist2(*(*it), w);

        //find first winner
        winner1 = (*it);
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {
            TNumber dist = dist2(*(*it), w);
            if (dist<minDist) {
                minDist = dist;
                winner1 = (*it);
            }
        }

        if (Mesh<TNode>::meshNodeSet.size()<2) {
            winner2 = NULL;
            return *this;
        }
        
        //find second winner
        it = Mesh<TNode>::meshNodeSet.begin();
        winner2 = (*it);
        
        if (winner2 == winner1) {
            it++;
            winner2 = (*it);
        }

        minDist = dist2(*(*it), w);
        TNode* distWinner = NULL;
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {
            if (*it!=winner1) {
                TNumber dist = dist2(*(*it), w);
                if (dist<minDist) {
                    minDist = dist;
                    winner2 = (*it);
                    if (wdist(*winner1, *(*it)) <= minwd)
                        distWinner = winner2;
                }
            }
        }

        if (distWinner!=NULL)
            winner2 = distWinner;

        return *this;
    }

    bool isNoise(const TVector &w) {
        TNode *winner = getWinner(w);
        double a = activation(*winner, w);
        return (a<a_t);
    }

    void resetToDefault(int dimw = 2) {
        GDSSOMC::dimw = dimw;
        step = 0;
        nodesLeft = 1;

        maxNodeNumber = 100;
        e_b = 0.05;
        e_n = 0.0006;
        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;
        nodesCounter = 2;

        destroyMesh();
        TVector v(dimw);
        v.random();
        TVector wNew(v);
        createNode(0, wNew);
    }

    void reset(int dimw) {
        GDSSOMC::dimw = dimw;
        step = 0;
        nodesLeft = 1;

        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;
        nodesCounter = 2;

        destroyMesh();
        TVector v(dimw);
        v.random();
        TVector wNew(v);
        createNode(0, wNew);
    }

    void reset(void) {
        reset(dimw);
    }

    GDSSOMC(int dimw) {
        resetToDefault(dimw);
    };

    ~GDSSOMC() {
    }

    template<class Number> GDSSOMC& outputCentersDs(MatMatrix<Number> &m) {
        using namespace std;

        uint wSize = (*meshNodeSet.begin())->ds.size();
        uint meshNodeSetSize = meshNodeSet.size();
        m.size(meshNodeSetSize, wSize);

        int i = 0;
        typename TPNodeSet::iterator it;
        for (it = meshNodeSet.begin(); it != meshNodeSet.end(); it++) {
            for (uint j = 0; j < wSize; j++)
                m[i][j] = (*it)->ds[j];
            i++;
        }

        return *this;
    }
};

#endif /* GDSSOMC_H_ */
