/* 
 * File:   LARFDSSOM.h
 * Author: daniel
 *
 * Created on 21 de Fevereiro de 2009, 19:26
 */

#ifndef _LARFDSSOM_H
#define _LARFDSSOM_H

#define qrt(x) ((x)*(x))

#include <set>
#include <map>
#include <vector>
#include <iomanip>
#include "Mesh.h"
#include "MatUtils.h"
#include "MatVector.h"
#include "MatMatrix.h"
#include "DSNode.h"

#include "SOM.h"

using namespace std;

class LARFDSNode;

class LARFDSConnection : public Connection<LARFDSNode> {
public:

    int age;

    LARFDSConnection(TNode *node0, TNode *node1) : Connection<LARFDSNode>(node0, node1), age(0) {
    }
};

class LARFDSNode : public DSNode {
public:

    typedef LARFDSConnection TConnection;
    typedef std::map<LARFDSNode*, TConnection*> TPNodeConnectionMap; //Para mapeamento local dos n�s e conex�es ligadas a this
    TPNodeConnectionMap nodeMap;

    int d; //Contador de vitórias do nó

    inline int neighbors() {
        return nodeMap.size();
    }

    LARFDSNode(int idIn, const TVector &v) : DSNode(idIn, v), d(0) {
    };

    ~LARFDSNode() {
    };
};

class LARFDSSOM : public SOM<LARFDSNode> {
public:
    uint maxNodeNumber;
    int d_max;
    int dimension;

    TNumber a_t; //Limiar de atividade
    TNumber epsilon; //Modulador da taxa de aprendizagem
    TNumber ho_f; //Taxa de aprendizagem
    TNumber dsbeta; //Taxa histórico
    TNumber epsilon_ds; //Relevância mínima


    //Fun��es utilizadas no treinamento

    inline float dist(const LARFDSNode &node, const TVector &w) {
        float distance = 0;

        for (uint i = 0; i < w.size(); i++) {
            distance += node.ds[i] * qrt((w[i] - node.w[i]));
        }

        //return sqrt(distance/(node.ds.sum()+ 0.0000001));
        return distance;
    }

    inline float distEucl(const LARFDSNode &node, const TVector &w) {
        float distance = 0;

        for (uint i = 0; i < w.size(); i++) {
            distance += qrt((w[i] - node.w[i]));
        }

        //return sqrt(distance/(node.ds.sum()+ 0.0000001));
        return distance;
    }

    inline float activation(TNode &node, const TVector &w, TNumber r) {

        float distance = 0;

        for (uint i = 0; i < w.size(); i++) {
            distance += node.ds[i] * qrt((w[i] - node.w[i]));
        }

        return exp(-dist(node, w)) / r;
        //return (node.ds.sum() / (node.ds.sum() + distance + 0.0000001))/r;
    }

    inline void updateDimensionTo(int newDimension) {
        //Fazer atualizacao do Tamanho_da_entrada da rede
        //Levando em consideração 12 característcas
        int somTam = this->size();
        LARFDSNode* nodoNow;
        MatVector<float> tempVector, tempVector_a;
        nodoNow = this->getFirstNode();
        int sizeTemp = (12 * newDimension) - nodoNow->w.size();
        tempVector.size(sizeTemp);
        tempVector_a.size(sizeTemp);
        tempVector.fill(0);
        

        for (int i = 0; i < somTam; i++) {

            nodoNow->w.concat(tempVector); //Aumentando o vetor de prototipos preenchendo com 0
            nodoNow->ds.concat(tempVector); //Aumentando o vetor de relevancias preenchendo com 0

            //Aumentando o vetor de distancias preenchendo com o valor maximo
            tempVector_a.fill(nodoNow->a.max());
            nodoNow->a.concat(tempVector_a);

            if (i < somTam - 1) {
                nodoNow = this->getNextNode();
            }
        }

    }

    inline void updateNode(TNode &node, const TVector &w) {

        //update averages
        for (uint i = 0; i < node.a.size(); i++) {
            //update neuron weights
            float distance = fabs(w[i] - node.w[i]);
            node.a[i] = dsbeta * distance + (1 - dsbeta) * node.a[i];
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

        /*
        float sumds = node.ds.sum();
        for (uint i = 0; i < node.a.size(); i++)
            node.ds[i] = node.ds[i] / sumds;
        /**/

        //update w
        TNumber ho;
        if (node.d <= d_max) {
            ho = epsilon * pow(ho_f, (double) (node.d / d_max));
        } else {
            ho = epsilon*ho_f;
        }

        node.w = node.w + ho * (w - node.w);
    }

    inline LARFDSSOM& updateMap(const TVector &w) {

        using namespace std;

        TNode *winner1 = 0;
        TNode *winner2 = 0;

        //Passo 3 : encontra os dois n�s vencedores
        getWinners(w, winner1, winner2); //winner
        winner1->d++;

        //Passo 4: Se os vencedores n�o est�o conectados conecta eles e com uma
        // aresta de age igual a zero
        if (!isConnected(winner1, winner2)) {
            //Se os n�s n�o est�o conectado cria uma conex�o entre os n�s com age igual a zero
            connect(winner1, winner2);
        }

        //Passo 5: Calcula o campo receptivo
        TNumber r = distEucl(*winner1, winner2->w);

        //Passo 6: Calcula a atividade do nó vencedor

        TNumber a = activation(*winner1, w, r); //DS activation

        /*Passo 6: Se a atividade for menor que o limiar de atividade e o
         * contador de disparo for menor que o limiar de disparo então
         *um novo nó pode ser adicionado entre o neurônio vencedor e o segundo
         *vencedor;
         */

        if ((a < a_t) && (meshNodeSet.size() < maxNodeNumber)) {

            TVector wNew(w);
            TNode *nodeNew = createNode(0, wNew);
            nodeNew->generation = nodeNew->w.size();
            //disconnect(winner1, winner2);
            TNumber d12 = dist(*winner1, winner2->w);
            TNumber d1New = dist(*winner1, w);
            TNumber d2New = dist(*winner2, w);

            if ((d12 < d1New) && (d12 < d2New)) {
                //Não disconecta wiiner1 de winner2, pois já estão com a menor distância
                if (d1New < d2New)//Conecta a segunda menor distância
                {
                    connect(winner1, nodeNew);
                } else {
                    connect(winner2, nodeNew);
                }
            } else
                if ((d1New < d12) && (d1New < d2New)) {
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

        } else {
            //Passo 7: Se nenhum nó for adicionado adapte a posição do vencedor
            //e dos seus vizinhos

            //Passo 7.1:Atualiza o nó vencedor
            updateNode(*winner1, w);

            /* Atualização dos vizinhos
            TPNodeConnectionMap &nodeMap = winner1->nodeMap;
            TPNodeConnectionMap::iterator it;
            for (it = nodeMap.begin(); it != nodeMap.end(); it++) {
                //Passo 7.2: Atualiza o peso dos vizinhos
                TNode* node = it->first;

                updateNode(*node, w);

                //Passo 8: Incrementa a age de todas as conex�es do n� vencedor
                it->second->age++;
            }
             */
        }
        /*
        //Passo 9:Remove todos os n�s sem conex�o
        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {
            if ((*itMesh)->neighbors() == 0) {
                eraseNode((*itMesh));
                itMesh = meshNodeSet.begin();
                cout << "Nodo removida\n";
            } else
                itMesh++;
        }
         */
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

    inline LARFDSSOM& getWinners(const TVector &w, TNode* &winner1, TNode* &winner2) {
        TPNodeSet::iterator it = Mesh<TNode>::meshNodeSet.begin();
        TNumber temp = 0;
        TNumber d2, d1 = dist(*(*it), w);
        winner1 = (*it);
        it++;
        if (dist(*(*it), w) < d1) {
            winner2 = winner1;
            d2 = d1;
            winner1 = (*it);
            d1 = dist(*(*it), w);
        } else {
            winner2 = (*it);
            d2 = dist(*(*it), w);
        }
        it++;
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {
            temp = dist(*(*it), w);
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

        return *this;
    }

    /*
     *Seleciona dois pontos iniciais do conjunto de dados
     */
    inline LARFDSSOM& initialize(const TMatrix &m) {
        destroyMesh();
        data = m;
        int vSize = data.cols();
        int N = data.rows();
        TVector v0(vSize);
        TVector v1(vSize);
        TNumber r;
        TNumber a;
        for (uint j = 0; j < vSize; j++)
            v0[j] = data[0][j];

        for (int i = 1; i < N; i++) {

            for (uint j = 0; j < vSize; j++)
                v1[j] = data[i][j];

            r = v0.dist(v1);
            a = exp(-r) / r;

            if (a < a_t)
                break;
        }

        makeBinaryGrid(v0, v1);
        return *this;
    }

    void resetToDefault(int dimw = 2) {
        LARFDSSOM::dimw = dimw;
        maxNodeNumber = 1000;
        d_max = 100;
        a_t = 3; //Limiar de atividade
        epsilon = 0.4; //Modulador da taxa de aprendizagem
        ho_f = 0.06; //Taxa de aprendizagem

        step = 0;
        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;
        TVector v0(dimw), v1(dimw);
        v0.random();
        v1.random();
        destroyMesh();
        makeBinaryGrid(v0, v1);
    }

    void reset(int dimw) {
        LARFDSSOM::dimw = dimw;
        step = 0;
        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;
        TVector v0(dimw), v1(dimw);
        v0.random();
        v1.random();
        destroyMesh();
        makeBinaryGrid(v0, v1);
    }

    void reset() {
        reset(dimw);
    }

    LARFDSSOM(int dimw) {
        using namespace std;
        resetToDefault(dimw);
    };

    ~LARFDSSOM() {

    }

    template<class Number> LARFDSSOM& outputCentersDs(MatMatrix<Number> &m) {
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


#endif /* _LARFDSSOM_H */

