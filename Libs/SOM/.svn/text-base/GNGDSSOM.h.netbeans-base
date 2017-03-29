/*
 * GNGDSSOM.h
 *
 *  Created on: 10/02/2009
 *      Author: daniel
 */

#ifndef GNGDSSOM_H_
#define GNGDSSOM_H_

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

class GNGDSNode;

class GNGDSConnection : public Connection<GNGDSNode> {
public:
    int age;

    GNGDSConnection(TNode *node0, TNode *node1) : Connection<GNGDSNode>(node0, node1), age(0) {
    }
};

class GNGDSNode : public DSNode {
public:

    typedef GNGDSConnection TConnection;
    typedef std::map<GNGDSNode*, TConnection*> TPNodeConnectionMap; //Para mapeamento local dos n�s e conex�es ligadas a this
    float erro;
    TPNodeConnectionMap nodeMap;

    inline int neighbors() {
        return nodeMap.size();
    }

    GNGDSNode(int idIn, const TVector &v) : DSNode(idIn, v), erro(0) {
    };

    ~GNGDSNode() {
    };
};

class GNGDSSOM : public SOM<GNGDSNode> {
public:
    int age_max;
    int step_max;
    uint maxNodeNumber;
    float alfa;
    float beta;
    float e_b;
    float e_n;
    int nodesCounter;
    TNumber dsbeta; //Taxa de aprendizagem
    TNumber epsilon_ds; //Taxa de aprendizagem

    inline float activation(const TNode &node, const TVector &w) {

        float distance = 0;

        for (uint i = 0; i < w.size(); i++) {
            distance += node.ds[i] * qrt((w[i] - node.w[i]));
        }

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

    inline void updateNode(TNode &node, const TVector &w, TNumber e) {

        //update averages
        for (uint i = 0; i < node.a.size(); i++) {
            //update neuron weights
            float distance = fabs(w[i] - node.w[i]);
            node.a[i] = dsbeta*e * distance + (1 - dsbeta*e) * node.a[i];
        }

        float max = node.a.max();

        //update neuron ds weights
        for (uint i = 0; i < node.a.size(); i++) {
            if (max != 0)
                node.ds[i] = 1 - (node.a[i] / max);
            else
                node.ds[i] = 1;

            if (node.ds[i] < epsilon_ds)
                node.ds[i] = epsilon_ds;
        }

        /* max
        float maxds = node.ds.max();
        for (uint i = 0; i < node.a.size(); i++)
            node.ds[i] = node.ds[i]/maxds;
        /**/

        /* sum
        float sum = node.ds.sum();
        for (uint i = 0; i < node.a.size(); i++)
            node.ds[i] = node.ds[i]/sum;
        /**/

        //Passo 5: Adiciona o erro ao n� vencedor
        node.erro += dist2(node, w); //TODO: testar com distância ponderada

        //Passo 6.1: Atualiza o peso do vencedor
        //Atualiza o n� vencedor
        node.w = node.w + e * (w - node.w);
    }

    GNGDSSOM& updateMap(const TVector &w) {

        using namespace std;
        //static int counter;
        //cout << "atualiza��es do mapa=" << counter++ << endl;
        //if(counter==100)
        TNode *winner1 = 0;
        TNode *winner2 = 0;

        //Passo 3 : encontra os dois n�s vencedores
        getWinners(w, winner1, winner2); //winner
        TPNodeConnectionMap &nodeMap = winner1->nodeMap;

        //Passo 4: Se os vencedores n�o est�o conectados conecta eles e com uma
        // aresta de age igual a zero
        if (!isConnected(winner1, winner2)) {
            //Se os n�s n�o est�o conectado cria uma conex�o entre os n�s com age igual a zero
            connect(winner1, winner2);
        } else {
            //Se os n�s j� est�o conectados reseta o age da connex�o que ligas os dois n�s
            winner1->nodeMap[winner2]->age = 0;
            //winner2->nodeMap[winner1]->age=0;
        }

        //Passo 5 e 6.1: Adiciona o erro ao n� vencedor e
        // Atualiza o peso do vencedor
        updateNode(*winner1, w, e_b);

        TPNodeConnectionMap::iterator it;
        for (it = nodeMap.begin(); it != nodeMap.end(); it++) {
            //Passo 6.2: Atualiza o peso dos vizinhos
            TNode* node = it->first;
            updateNode(*node, w, e_n);

            //Passo 7: Incrementa a age de todas as conex�es do n� vencedor
            it->second->age++;
        }

        //Passo 8:Remove todas as conex�es com age maior que um limiar
        it = nodeMap.begin();
        while (it != nodeMap.end())//for(it=nodeMap.begin();it!=nodeMap.end();it++)
        {
            if ((it->second->age) > age_max) {
                //if(meshConnectionSet.size()>2)

                disconnect(it->second);
                it = nodeMap.begin();

                //else
                //break;
            } else
                it++;
        }

        //Remove todos os n�s sem conex�o
        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {
            if ((*itMesh)->neighbors() == 0) {
                eraseNode((*itMesh));
                itMesh = meshNodeSet.begin();
            } else
                itMesh++;
        }

        //Passo 9:Se o n�mero de passo for multiplo de um valor gama insere um n�
        if (step >= step_max) {
            step = 0;

            if (meshNodeSet.size() < maxNodeNumber) {
                TPNodeSet::iterator itNode = meshNodeSet.begin();
                TNode *maxErrorNode = (*itNode);
                itNode++;
                for (; itNode != meshNodeSet.end(); itNode++) {
                    if ((*itNode)->erro > maxErrorNode->erro) {
                        maxErrorNode = (*itNode);
                    }
                }

                TPNodeConnectionMap::iterator itMap = maxErrorNode->nodeMap.begin();
                TNode *maxErrorNeighbor = itMap->first;
                itMap++;
                for (; itMap != maxErrorNode->nodeMap.end(); itMap++) {
                    if (itMap->first->erro > maxErrorNeighbor->erro) {
                        maxErrorNeighbor = itMap->first;
                    }
                }

                disconnect(maxErrorNode, maxErrorNeighbor);
                TVector wNew = maxErrorNode->w + maxErrorNeighbor->w;
                wNew.div(2);
                TNode *nodeNew = createNode(nodesCounter++, wNew);
                connect(maxErrorNode, nodeNew);
                connect(maxErrorNeighbor, nodeNew);

                maxErrorNode->erro -= alfa * maxErrorNode->erro;
                maxErrorNeighbor->erro -= alfa * maxErrorNeighbor->erro;
                nodeNew->erro = (maxErrorNode->erro + maxErrorNeighbor->erro) / 2;
            }
        }


        //Passo 10: decrementa o erro de todas as unidades por um fator de beta
        TPNodeSet::iterator itNode;
        for (itNode = meshNodeSet.begin(); itNode != meshNodeSet.end(); itNode++) {
            (*itNode)->erro -= beta * (*itNode)->erro;
        }

        step++;
        return *this;
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

    inline GNGDSSOM& getWinners(const TVector &w, TNode* &winner1, TNode* &winner2) {
        TPNodeSet::iterator it = Mesh<TNode>::meshNodeSet.begin();
        TNumber temp = 0;
        TNumber d2, d1 = dist2(*(*it), w);
        winner1 = (*it);
        it++;
        if (dist2(*(*it), w) < d1) {
            winner2 = winner1;
            d2 = d1;
            winner1 = (*it);
            d1 = dist2(*(*it), w);
        } else {
            winner2 = (*it);
            d2 = dist2(*(*it), w);
        }
        it++;
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {
            temp = dist2(*(*it), w);
            if (d1 > temp) {
                d2 = d1;
                d1 = temp;
                winner2 = winner1;
                winner1 = (*it);
            } else
                if (d2 > temp) {
                winner2 = (*it);
                d2 = temp;
            }
        }

        /*
        dbgOut(1) << "w" << w.toString() << endl << endl;
        
        dbgOut(1) << "w1:  " << winner1->w.toString() << endl;
        dbgOut(1) << "a1:  " << winner1->a.toString() << endl;
        dbgOut(1) << "ds1: " << winner1->ds.toString() << endl << endl;
        
        dbgOut(1) << "w2:  " << winner2->w.toString() << endl;
        dbgOut(1) << "a2:  " << winner2->a.toString() << endl;
        dbgOut(1) << "ds2: " << winner2->ds.toString() << endl << endl;
        /**/
        return *this;
    }

    void resetToDefault(int dimw = 2) {
        GNGDSSOM::dimw = dimw;
        step = 0;

        age_max = 1000;
        step_max = 500;
        alfa = 0.5;
        beta = 0.0005;
        maxNodeNumber = 100;
        e_b = 0.05;
        e_n = 0.0006;
        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;
        nodesCounter = 2;

        TVector v0(dimw), v1(dimw);
        v0.random();
        v1.random();
        destroyMesh();
        makeBinaryGrid(v0, v1);
    }

    void reset(int dimw) {
        GNGDSSOM::dimw = dimw;
        step = 0;

        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;
        nodesCounter = 2;

        TVector v0(dimw), v1(dimw);
        v0.random();
        v1.random();
        destroyMesh();
        makeBinaryGrid(v0, v1);

    }

    void reset(void) {
        reset(dimw);
    }

    GNGDSSOM(int dimw) {
        resetToDefault(dimw);
    };

    ~GNGDSSOM() {
    }

    template<class Number> GNGDSSOM& outputCentersDs(MatMatrix<Number> &m) {
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

#endif /* GNGDSSOM_H_ */
