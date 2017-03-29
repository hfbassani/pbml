/*
 * GDSSOMMWF.h
 *
 *  Created on: 10/02/2009
 *      Author: daniel
 */

#ifndef GDSSOMMWF_H_
#define GDSSOMMWF_H_

#include <set>
#include <map>
#include <vector>
#include "Mesh.h"
#include "MatUtils.h"
#include "MatVector.h"
#include "MatMatrix.h"
#include "SOM.h"
#include "DSNode.h"

#define qrt(x) ((x)*(x))

using namespace std;

class GDSNodeMWF;

class GDSConnectionMWF : public Connection<GDSNodeMWF> {
public:
    int age;

    GDSConnectionMWF(TNode *node0, TNode *node1) : Connection<GDSNodeMWF>(node0, node1), age(0) {
    }
};

class GDSNodeMWF : public DSNode {
public:

    typedef GDSConnectionMWF TConnection;
    typedef std::map<GDSNodeMWF*, TConnection*> TPNodeConnectionMap; //Para mapeamento local dos n�s e conex�es ligadas a this

    int wins;
    TPNodeConnectionMap nodeMap;
    TNumber act;

    inline int neighbors() {
        return nodeMap.size();
    }

    GDSNodeMWF(int idIn, const TVector &v) : DSNode(idIn, v), wins(0), act(0) {
    };

    ~GDSNodeMWF() {
    };
};

class GDSSOMMWF : public SOM<GDSNodeMWF> {
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
    bool hamming;
    
    int nodesLeft;
    TVector fixed;

    inline float activation(const TNode &node, const TVector &w) {

        float distance = 0;

        if (hamming) {
            for (uint i = 0; i < w.size(); i++) {
                if (fabs(w[i] - node.w[i])>0.5)
                      distance += node.ds[i];
            }
        }
        else {
            for (uint i = 0; i < w.size(); i++) {
                distance += node.ds[i] * qrt((w[i] - node.w[i]));
            }
        }

        //return (1/(distance + 0.0000001));
        return (node.ds.sum() / (node.ds.sum() + distance + 0.0000001));
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
            
            if (fixed[i]>=0) {
                node.ds[i] = fixed[i];
                continue;
            }
            
            if (max != 0)
                node.ds[i] = 1 - (node.a[i] / max);
            else
                node.ds[i] = 1;

//            if (node.ds[i] < 0.95*dsa)
//                node.ds[i] = epsilon_ds;
            
            if (node.ds[i] < epsilon_ds)
                node.ds[i] = epsilon_ds;
        }

        //Passo 6.1: Atualiza o peso do vencedor
        //Atualiza o nó vencedor
        node.w = node.w + e * (w - node.w);
    }

    GDSSOMMWF& updateConnections(TNode *node) {
        
        TPNodeSet::iterator itMesh = meshNodeSet.begin();
            
        while (itMesh != meshNodeSet.end()) {
            if (*itMesh != node) {
                if (wdist(*node, *(*itMesh))<minwd) {
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
    
    GDSSOMMWF& updateAllConnections() {

        //Conecta todos os nodos semelhantes
        TPNodeSet::iterator itMesh1 = meshNodeSet.begin();
        while (itMesh1 != meshNodeSet.end()) {
            TPNodeSet::iterator itMesh2 = meshNodeSet.begin();
            
            while (itMesh2 != meshNodeSet.end()) {
                if (*itMesh1!= *itMesh2) {
                    if (wdist(*(*itMesh1), *(*itMesh2))<minwd) {
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

    GDSSOMMWF& removeLoosers() {

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
                //dbgOut(1) << (*itMesh)->getId() << ": " << (*itMesh)->wins << "\t<\t" << step*lp << endl;
                eraseNode((*itMesh));
                itMesh = meshNodeSet.begin();
            } else {
                //dbgOut(1) << (*itMesh)->getId() << ": " << (*itMesh)->wins << "\t>=\t" << step*lp << endl;
                itMesh++;
            }
        }

        return *this;
    }

    GDSSOMMWF& finishMap() {

        do {
            resetWins();
            maxNodeNumber = meshNodeSet.size();
            trainning(age_wins+1);
            //*
            resetWins();

            TVector v;
            for (int i=0; i<data.rows(); i++) {
                data.getRow(i, v);
                TNode *winner = getWinner(v);
                if (activation(*winner, v)>= a_t) {
                    winner->wins++;
                    //step++;
                }
            }

            step = data.rows();
            //if (step==0) step = 1;

            int prefMeshSize = meshNodeSet.size();
            removeLoosers();
            updateAllConnections();            
            //*/
            if (maxNodeNumber == meshNodeSet.size() || meshNodeSet.size()==1)
                break;
        } while (true);
        
        return *this;
    }

    GDSSOMMWF& resetWins() {

        //Remove os perdedores
        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {
             (*itMesh)->wins = 0;
             itMesh++;
        }

        step = 0;

        return *this;
    }

    GDSSOMMWF& updateMap(const TVector &w) {

        using namespace std;
        TNode *winner1 = 0;

        //Passo 3 : encontra os dois nós vencedores
        winner1 = getWinner(w); //winner

        //Passo 6: Calcula a atividade do nó vencedor
        TNumber a = activation(*winner1, w); //DS activation

        //Se a ativação obtida pelo primeiro vencedor for menor que o limiar
        //e o limite de nodos não tiver sido atingido

        if ((a < a_t) && (meshNodeSet.size() < maxNodeNumber)) {
            //dbgOut(2) << a << "\t<\t" << a_t << endl;
            //Cria um novo nodo no local do padrão observado
            TVector wNew(w);
            TNode *nodeNew = createNode(0, wNew);
            nodeNew->wins = lp*step;

            //Conecta o nodo
            updateConnections(nodeNew);

        } else if (a >= a_t) { // caso contrário
            winner1->wins++;
            
            // Atualiza o peso do vencedor
            updateNode(*winner1, w, e_b);

            //Passo 6.2: Atualiza o peso dos vizinhos
            TPNodeConnectionMap::iterator it;
            for (it = winner1->nodeMap.begin(); it != winner1->nodeMap.end(); it++) {            
                TNode* node = it->first;
                updateNode(*node, w, e_n);
            }
        }

        //Passo 9:Se atingiu age_wins
        if (step >= age_wins) {

            int size = meshNodeSet.size();
            //remove os perdedores
            removeLoosers();
            dbgOut(1) << size << "\t->\t" << meshNodeSet.size() << endl;
            //reseta o número de vitórias
            //resetWins();
            //Passo 8.2:Adiciona conexões entre nodos semelhantes
            updateAllConnections();
            step = 0;
        }

        step++;
        return *this;
    }

    virtual TNode *getFirstWinner(const TVector &w){
        TNode *winner = 0;
        TNumber temp = 0;
        TNumber d = dist(*(*Mesh<TNode>::meshNodeSet.begin()), w);
        winner = (*Mesh<TNode>::meshNodeSet.begin());
        winner->act = activation(*winner, w);

        TPNodeSet::iterator it;
        it = Mesh<TNode>::meshNodeSet.begin();
        it++;
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {
            (*it)->act = activation(*(*it), w);
            temp = dist(*(*it), w);
            if (d > temp) {
                d = temp;
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
        
        if (winnerAct < a_t)
            return NULL;

        return winner;
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

    inline GDSSOMMWF& getWinners(const TVector &w, TNode* &winner1, TNode* &winner2) {
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
        
        //find second winner
        it = Mesh<TNode>::meshNodeSet.begin();
        winner2 = (*it);
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
        GDSSOMMWF::dimw = dimw;
        step = 0;
        nodesLeft = 1;

        maxNodeNumber = 100;
        e_b = 0.05;
        e_n = 0.0006;
        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;
        nodesCounter = 2;
        hamming = false;
        
        fixed.size(dimw);
        fixed.fill(-1);

        destroyMesh();
        TVector v(dimw);
        v.random();
        TVector wNew(v);
        createNode(0, wNew);
    }

    void reset(int dimw) {
        GDSSOMMWF::dimw = dimw;
        step = 0;
        nodesLeft = 1;

        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;
        nodesCounter = 2;

        fixed.size(dimw);
        fixed.fill(-1);
        
        destroyMesh();
        TVector v(dimw);
        v.random();
        TVector wNew(v);
        createNode(0, wNew);
    }

    void reset(void) {
        reset(dimw);
    }

    GDSSOMMWF(int dimw) {
        resetToDefault(dimw);
    };

    ~GDSSOMMWF() {
    }

    template<class Number> GDSSOMMWF& outputCentersDs(MatMatrix<Number> &m) {
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

#endif /* GDSSOMMWF_H_ */
