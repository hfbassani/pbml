/* 
 * File:   GWRSOM.h
 * Author: daniel
 *
 * Created on 21 de Fevereiro de 2009, 19:26
 */

#ifndef _GWRSOM_H
#define	_GWRSOM_H

#include <set>
#include <map>
#include <vector>
#include "Mesh.h"
#include "MatUtils.h"
#include "MatVector.h"
#include "MatMatrix.h"

#include "SOM.h"
#include "NodeW.h"


class GWRNode;

class GWRConnection : public Connection<GWRNode> {
public:

    int age;

    GWRConnection(TNode *node0, TNode *node1) : Connection<GWRNode>(node0, node1), age(0) {
    }
};

class GWRNode : public NodeW {
public:

    typedef GWRConnection TConnection;
    typedef std::map<GWRNode*, TConnection*> TPNodeConnectionMap; //Para mapeamento local dos n�s e conex�es ligadas a this
    TPNodeConnectionMap nodeMap;
    TNumber h; //Contador de disparo

    inline int neighbors() {
        return nodeMap.size();
    }

    GWRNode(int idIn, const TVector &v) : NodeW(idIn, v), h(0) {
    };

    ~GWRNode() {
    };

};

class GWRSOM : public SOM<GWRNode> {
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

      //Fun��es utilizadas no treinamento

    inline GWRSOM& updateMap(const TVector &w) {

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
        TNumber a = exp(-winner1->w.dist(w));

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
            winner1->w = winner1->w + e_b * winner1->h * (w - winner1->w);

            //Atualização dos vizinhos
            TPNodeConnectionMap::iterator it;
            for (it = nodeMap.begin(); it != nodeMap.end(); it++) {
                //Passo 7.2: Atualiza o peso dos vizinhos
                TNode* node = it->first;
                node->w = node->w + e_n * node->h * (w - node->w);

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
                        TPConnectionSet::iterator itCon=meshConnectionSet.begin();
        		while(itCon!=meshConnectionSet.end())
        		{
        			if(((*itCon)->age)>age_max)
        			{
                			disconnect((*itCon));
        				itCon=meshConnectionSet.begin();
        			}
        			else
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

    GWRSOM(int dimw) {
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
        GWRSOM::dimw = dimw;
        step = 0;
        age_max = 1000;        
        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;

        TVector v0(dimw), v1(dimw);
        v0.random();
        v1.random();
        destroyMesh();
        makeBinaryGrid(v0, v1);
    };
    
    ~GWRSOM() {

    }

};


#endif	/* _GWRSOM_H */

