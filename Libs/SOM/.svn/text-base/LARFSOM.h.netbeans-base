/* 
 * File:   LARFSOM.h
 * Author: daniel
 *
 * Created on 21 de Fevereiro de 2009, 19:26
 */

#ifndef _LARFSOM_H
#define	_LARFSOM_H

#include <set>
#include <map>
#include <vector>
#include "Mesh.h"
#include "MatUtils.h"
#include "MatVector.h"
#include "MatMatrix.h"

#include "SOM.h"
#include "NodeW.h"

class LARFNode;

class LARFConnection : public Connection<LARFNode> {
public:

    int age;

    LARFConnection(TNode *node0, TNode *node1) : Connection<LARFNode>(node0, node1), age(0) {
    }
};

class LARFNode : public NodeW {
public:

    typedef LARFConnection TConnection;
    typedef std::map<LARFNode*, TConnection*> TPNodeConnectionMap; //Para mapeamento local dos n�s e conex�es ligadas a this
    TPNodeConnectionMap nodeMap;
    int d; //Contador de vitórias do nó

    inline int neighbors() {
        return nodeMap.size();
    }

    LARFNode(int idIn, const TVector &v) : NodeW(idIn, v), d(0) {
    };

    ~LARFNode() {
    };

};

class LARFSOM : public SOM<LARFNode> {
public:
    uint maxNodeNumber;
    int d_max;

    TNumber a_t; //Limiar de atividade
    TNumber epsilon; //Modulador da taxa de aprendizagem
    TNumber ho_f; //Taxa de aprendizagem


    //Fun��es utilizadas no treinamento

    inline LARFSOM& updateMap(const TVector &w) {

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
        TNumber r = winner1->w.dist(winner2->w);

        //Passo 6: Calcula a atividade do nó vencedor
        TNumber a = exp(-winner1->w.dist(w)) / r;

        /*Passo 6: Se a atividade for menor que o limiar de atividade e o
         * contador de disparo for menor que o limiar de disparo então
         *um novo nó pode ser adicionado entre o neurônio vencedor e o segundo
         *vencedor;
         */

        if ((a < a_t)&&(meshNodeSet.size()<maxNodeNumber)) {

            TVector wNew(w);
            TNode *nodeNew = createNode(0, wNew);

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
            //Passo 7: Se nenhum nó for adicionado adapti a posição do vencedor
            //e dos seus vizinhos

            //Passo 7.1:Atualiza o nó vencedor
            TNumber ho;
            if (winner1->d <= d_max) {
                ho = epsilon * pow(ho_f, (double) (winner1->d / d_max));
            } else {
                ho = epsilon*ho_f;
            }
            winner1->w = winner1->w + ho * (w - winner1->w);

            //Atualização dos vizinhos
            //            TPNodeConnectionMap::iterator it;
            //            for (it = nodeMap.begin(); it != nodeMap.end(); it++) {
            //                //Passo 7.2: Atualiza o peso dos vizinhos
            //                TNode* node = it->first;
            //
            //                if (node->d <= d_max) {
            //                    ho = epsilon * pow(ho_f, (double) (node->d / d_max));
            //                } else {
            //                    ho = epsilon*ho_f;
            //                }
            //
            //                node->w = node->w + ho * (w - node->w);
            //
            //                //Passo 8: Incrementa a age de todas as conex�es do n� vencedor
            //                it->second->age++;
            //            }
        }

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

        step++;
        return *this;
    }

    /*
     *Seleciona dois pontos iniciais do conjunto de dados
     */
    inline LARFSOM& initialize(const TMatrix &m) {
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
        LARFSOM::dimw = dimw;
        maxNodeNumber=1000;
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
        LARFSOM::dimw = dimw;
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

    LARFSOM(int dimw) {
        using namespace std;
        resetToDefault(dimw);
    };

    ~LARFSOM() {

    }

    TPNodeSet::iterator it;
    TNode *getFirstNode() {
        it = meshNodeSet.begin();
        return (*it);
    }

    TNode *getNextNode() {
        it++;
        return (*it);
    }

    bool finished() {
       return (it==meshNodeSet.end());
    }
};


#endif	/* _LARFSOM_H */

