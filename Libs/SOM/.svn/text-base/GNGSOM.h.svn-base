/*
 * GNGSOM.h
 *
 *  Created on: 10/02/2009
 *      Author: daniel
 */

#ifndef GNGSOM_H_
#define GNGSOM_H_

#include <set>
#include <map>
#include <vector>
#include "Mesh.h"
#include "MatUtils.h"
#include "MatVector.h"
#include "MatMatrix.h"
#include "SOM.h"
#include "NodeW.h"


class GNGNode;

class GNGConnection : public Connection<GNGNode> {
public:
    int age;

    GNGConnection(TNode *node0, TNode *node1) : Connection<GNGNode>(node0, node1), age(0) {
    }
};

class GNGNode : public NodeW {
public:

    typedef GNGConnection TConnection;
    typedef std::map<GNGNode*, TConnection*> TPNodeConnectionMap; //Para mapeamento local dos n�s e conex�es ligadas a this
    float erro;
    TPNodeConnectionMap nodeMap;

    inline int neighbors() {
        return nodeMap.size();
    }

    GNGNode(int idIn, const TVector &v) : NodeW(idIn, v), erro(0) {
    };

    ~GNGNode() {
    };

};

class GNGSOM : public SOM<GNGNode> {
public:


    int age_max;
    int step_max;
    uint maxNodeNumber;
    float alfa;
    float beta;
    float e_b;
    float e_n;
    int nodesCounter;

    GNGSOM& updateMap(const TVector &w) {

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

        //Passo 5: Adiciona o erro ao n� vencedor
        winner1->erro += winner1->w.dist2(w);

        //Passo 6.1: Atualiza o peso do vencedor
        //Atualiza o n� vencedor
        winner1->w = winner1->w + e_b * (w - winner1->w);

        TPNodeConnectionMap::iterator it;
        for (it = nodeMap.begin(); it != nodeMap.end(); it++) {
            //Passo 6.2: Atualiza o peso dos vizinhos
            TNode* node = it->first;
            node->w = node->w + e_n * (w - node->w);

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

    void resetToDefault(int dimw = 2) {
        GNGSOM::dimw = dimw;
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
        GNGSOM::dimw = dimw;
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

    GNGSOM(int dimw) {
        resetToDefault(dimw);
    };

    ~GNGSOM() {

    }

};

#endif /* GNGSOM_H_ */
