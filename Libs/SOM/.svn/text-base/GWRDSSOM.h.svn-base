/* 
 * File:   GWRSOM.h
 * Author: daniel
 *
 * Created on 21 de Fevereiro de 2009, 19:26
 */

#ifndef _GWRDSSOM_H
#define	_GWRDSSOM_H

#include <set>
#include <map>
#include <vector>
#include "Mesh.h"
#include "MatUtils.h"
#include "MatVector.h"
#include "MatMatrix.h"
#include "DSNode.h"
#include "DebugOut.h"

#define qrt(x) ((x)*(x))
using namespace std;

class GWRDSNode;

class GWRDSConnection : public Connection<GWRDSNode> {
public:

    int age;

    GWRDSConnection(TNode *node0, TNode *node1) : Connection<GWRDSNode>(node0, node1), age(0) {
    }
};

class GWRDSNode : public DSNode {
public:

    typedef GWRDSConnection TConnection;
    typedef std::map<GWRDSNode*, TConnection*> TPNodeConnectionMap; //Para mapeamento local dos n�s e conex�es ligadas a this
    TPNodeConnectionMap nodeMap;
    TNumber h; //Contador de disparo

    inline int neighbors() {
        return nodeMap.size();
    }

    GWRDSNode(int idIn, const TVector &v) : DSNode(idIn, v), h(0) {
    };

    ~GWRDSNode() {
    };

};

class GWRDSSOM : public SOM<GWRDSNode> {
public:

    int age_max;
    uint maxNodeNumber;

    TNumber a_t; //Limiar de atividade
    TNumber h_t; //Limiar de disparo
    TNumber e_b;
    TNumber e_n;
    TNumber h0;
    TNumber iStrength;
    TNumber a_b;
    TNumber a_n;
    TNumber tau_b;
    TNumber tau_n;
    TNumber dsbeta; //Taxa de aprendizagem
    TNumber epsilon_ds; //Taxa de aprendizagem

    //Fun��es utilizadas no treinamento

    inline float dist2(const GWRDSNode &node, const TVector &w) {
        float distance = 0;

        for (uint i = 0; i < w.size(); i++) {
            distance += node.ds[i] * qrt((w[i] - node.w[i]));
        }

        return sqrt(distance/(node.ds.sum()+ 0.0000001));
        //return distance;
    }

    inline float dist(const GWRDSNode &node, const TVector &w) {
        return sqrt(dist2(node, w));
    }

    inline float activation(TNode &node, const TVector &w) {

        float distance = 0;

        for (uint i = 0; i < w.size(); i++) {
            distance += node.ds[i] * qrt((w[i] - node.w[i]));
        }

        return exp(-dist(node, w));
        //return (node.ds.sum() / (node.ds.sum() + distance + 0.0000001));
    }

    inline void updateNode(TNode &node, const TVector &w, TNumber e) {

        //update averages
        for (uint i = 0; i < node.a.size(); i++) {

            float distance = fabs(w[i] - node.w[i]);
            node.a[i] = e*dsbeta * distance + (1 - e*dsbeta) * node.a[i];
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
        
        //Atualizar nodo
        node.w = node.w + e * node.h * (w - node.w);
    }

    inline GWRDSSOM& updateMap(const TVector &w) {

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

        //Passo 5: Calcula a atividade do nó vencedor
        TNumber a = activation(*winner1, w); //DS activation

        /*Passo 6: Se a atividade for menor que o limiar de atividade e o
         * contador de disparo for menor que o limiar de disparo então
         *um novo nó pode ser adicionado entre o neurônio vencedor e o segundo
         *vencedor;
         */
        if ((a < a_t) && (winner1->h < h_t) && (meshNodeSet.size() < maxNodeNumber)) {
            disconnect(winner1, winner2);
            TVector wNew = winner1->w + w;
            wNew.div(2);
            TNode *nodeNew = createNode(0, wNew);
            connect(winner1, nodeNew);
            connect(winner2, nodeNew);
        } else {
            //Passo 7: Se nenhum nó for adicionado adapti a posição do vencedor
            //e dos seus vizinhos

            //Passo 7.1:Atualiza o nó vencedor
            updateNode(*winner1, w, e_b);

            //Atualização dos vizinhos
            TPNodeConnectionMap::iterator it;
            for (it = nodeMap.begin(); it != nodeMap.end(); it++) {
                //Passo 7.2: Atualiza o peso dos vizinhos
                TNode* node = it->first;

                updateNode(*node, w, e_n);

                //Passo 8: Incrementa a age de todas as conex�es do n� vencedor
                it->second->age++;
            }
        }

        //Passo 9:Reduzir o contador de disparo do nó vencedor e dos vizinhos
        winner1->h = h0 - (iStrength / a_b)*(1 - exp(-a_b * step / tau_b));

        TNumber h_i = h0 - (iStrength / a_n)*(1 - exp(-a_n * step / tau_n));
        TPNodeConnectionMap::iterator it;
        for (it = nodeMap.begin(); it != nodeMap.end(); it++) {
            it->first->h = h_i;
        }


        //Passo 10.1:Remove todas as conex�es com age maior que um limiar
        TPConnectionSet::iterator itCon = meshConnectionSet.begin();
        while (itCon != meshConnectionSet.end()) {
            if (((*itCon)->age) > age_max) {
                disconnect((*itCon));
                itCon = meshConnectionSet.begin();
            } else
                itCon++;
        }

        it = nodeMap.begin();
        while (it != nodeMap.end()) {
            if ((it->second->age) > age_max) {
                disconnect(it->second);
                it = nodeMap.begin();
            } else
                it++;
        }

        //Passo 10.2:Remove todos os n�s sem conex�o
        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {
            if ((*itMesh)->neighbors() == 0) {
                eraseNode((*itMesh));
                itMesh = meshNodeSet.begin();
            } else
                itMesh++;
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

    inline GWRDSSOM& getWinners(const TVector &w, TNode* &winner1, TNode* &winner2) {
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

    GWRDSSOM(int dimw) {
        using namespace std;
        step = 0;
        age_max = 1000;
        maxNodeNumber = 100;
        e_b = 0.05;
        e_n = 0.0006;
        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;

        TVector v0(dimw), v1(dimw);
        v0.random();
        v1.random();
        makeBinaryGrid(v0, v1);
    };

    void reset(int dimw) {
        GWRDSSOM::dimw = dimw;
        step = 0;
        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;

        TVector v0(dimw), v1(dimw);
        v0.random();
        v1.random();
        destroyMesh();
        makeBinaryGrid(v0, v1);
    };
    
    ~GWRDSSOM() {

    }

    template<class Number> GWRDSSOM& outputCentersDs(MatMatrix<Number> &m) {
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


#endif	/* _GWRDSSOM_H */

