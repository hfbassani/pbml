/*
 * VILMAP.h
 *
 *  Created on: 2014
 *      Author: hans
 */

#ifndef VILMAP_H_
#define VILMAP_H_

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

class GDSNodeMW;

class GDSConnectionMW : public Connection<GDSNodeMW> {
public:
    int age;

    GDSConnectionMW(TNode *node0, TNode *node1) : Connection<GDSNodeMW>(node0, node1), age(0) {
    }
};

class GDSNodeMW : public DSNode {
public:

    typedef GDSConnectionMW TConnection;
    typedef std::map<GDSNodeMW*, TConnection*> TPNodeConnectionMap; //Para mapeamento local dos n�s e conex�es ligadas a this

    int wins;
    TPNodeConnectionMap nodeMap;
    TNumber act;

    inline int neighbors() {
        return nodeMap.size();
    }

    GDSNodeMW(int idIn, const TVector &v) : DSNode(idIn, v), wins(0), act(0) {
    };

    ~GDSNodeMW() {
    };
};

class VILMAP : public SOM<GDSNodeMW> {
public:
    uint maxNodeNumber;
    float minwd;
    float e_b;
    float e_n;
    int nodesCounter;

    TNumber dsbeta; //Taxa de aprendizagem
    TNumber epsilon_ds; //Taxa de relevancia
    float age_wins; //period to remove nodes
    float lp; //remove percentage threshold
    float a_t;
    int d_min; //dimensão inicial
    int d_max; //dimensão máxima
    int nodesLeft;
    int nodeID;

    inline float activation(TNode &node, const TVector &w) {
        int end = 0;
        float tempDistance = 0;
        float distance = 0;
        node.index = 0;
        if (node.w.size() <= w.size()) {
            end = node.w.size();
            for (uint i = 0; i < end; i++) {
                distance += node.ds[i] * qrt((w[i] - node.w[i]));
                if (std::isnan(w[i]) || std::isnan(distance)) {
                    std::cout << i << " - Debug 1" << endl;
                }
            }
        } else {

            distance = 99999999;
            for (uint i = 0; i <= (node.w.size() - w.size()); i += 12) {
                tempDistance = 0;
                for (uint j = 0; j < w.size(); j++) {
                    //cout << i << " - " << j << " - " << tempDistance << " | ";
                    tempDistance += node.ds[i + j] * qrt((w[j] - node.w[i + j]));
                    if (std::isnan(w[j]) || std::isnan(tempDistance)) {
                        std::cout << i << " - Debug 2" << endl;
                    }

                }

                if (tempDistance < distance) {
                    distance = tempDistance;
                    node.index = 0;
                }
            }
        }


        //dbgOut(1) <<"N:" << node.w.size() << "\t" << "E:" << w.size();


        float sum = node.ds.sum();

        return (sum / (sum + distance + 0.0000001));
    }

    float getWinnerActivation(const TVector &w) {
        TNode* winner = getWinner(w);
        float a = activation(*winner, w);
        dbgOut(2) << winner->getId() << "\t" << a << endl;
        return a;
    }

    inline float dist2(TNode &node, const TVector &w) {
        /*float distance = 0;

        for (uint i = 0; i < w.size(); i++) {
            distance += node.ds[i] * qrt((w[i] - node.w[i]));
        }

        return distance / (node.ds.sum() + 0.0000001);*/
        return 1 / activation(node, w);
    }

    inline float dist(TNode &node, const TVector &w) {
        return sqrt(dist2(node, w));
    }

    inline float wdist(const TNode &node1, const TNode &node2) {
        float distance = 0;

        for (uint i = 0; i < node1.ds.size(); i++) {
            distance += qrt((node1.ds[i] - node2.ds[i]));
        }

        return sqrt(distance);
    }

    inline void updateNode(TNode &node, const TVector &w, TNumber e) {
        uint end = 0;
        if (node.w.size() < w.size()) {
            end = node.w.size();
        } else {
            end = w.size();
        }
        uint begin = 0;
        if (node.w.size() > w.size() && node.index != 0) {
            begin = node.index; //o nodo será atualizado começando na posição de maior ativação
            end = begin + w.size();
        }
        //update averages
        for (uint i = begin, t = 0; i < end; i++, t++) {
            //update neuron weights
            float distance = fabs(w[t] - node.w[i]);
            if (std::isnan(distance)) {
                distance = 1;
            }
            node.a[i] = e * dsbeta * distance + (1 - e * dsbeta) * node.a[i];
        }

        float max = node.a.max();
        float min = node.a.min();
        float average = node.a.mean();
        //float dsa = node.ds.mean();

        //update neuron ds weights
        for (uint i = begin; i < node.a.size(); i++) {
            if ((max - min) != 0) {
                //node.ds[i] = 1 - (node.a[i] - min) / (max - min);
                node.ds[i] = 1 / (1 + exp((node.a[i] - average) / (((max - min) * epsilon_ds) + 0.000001)));
            } else
                node.ds[i] = 1;
        }

        //Passo 6.1: Atualiza o peso do vencedor
        //Atualiza o nó vencedor
        for (uint i = begin, t = 0; i < end; i++, t++) {
            node.w = node.w[i] + e * (w[t] - node.w[i]);
        }

    }

    VILMAP& updateConnections(TNode *node) {

        TPNodeSet::iterator itMesh = meshNodeSet.begin();

        while (itMesh != meshNodeSet.end()) {
            if (*itMesh != node) {
                if (wdist(*node, *(*itMesh)) < minwd) {
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

    VILMAP& updateAllConnections() {

        //Conecta todos os nodos semelhantes
        TPNodeSet::iterator itMesh1 = meshNodeSet.begin();
        while (itMesh1 != meshNodeSet.end()) {
            TPNodeSet::iterator itMesh2 = meshNodeSet.begin();

            while (itMesh2 != meshNodeSet.end()) {
                if (*itMesh1 != *itMesh2) {
                    if (wdist(*(*itMesh1), *(*itMesh2)) < minwd) {
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

    VILMAP& removeLoosers() {

        //enumerateNodes();

        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {
            if (meshNodeSet.size() < 2)
                break;

            if ((*itMesh)->wins < step * lp) {
                //dbgOut(1) << (*itMesh)->getId() << ": " << (*itMesh)->wins << "\t<\t" << step*lp << endl;
                eraseNode((*itMesh));
                itMesh = meshNodeSet.begin();
            } else {
                //dbgOut(1) << (*itMesh)->getId() << ": " << (*itMesh)->wins << "\t>=\t" << step*lp << endl;
                itMesh++;
            }
        }

        //printWinners();
        return *this;
    }

    //*

    VILMAP& finishMap() {

        dbgOut(1) << "Finishing map..." << endl;
        do {
            resetWins();
            maxNodeNumber = meshNodeSet.size();
            trainning(age_wins);

            resetWins();

            TVector v;
            for (int i = 0; i < data.rows(); i++) {
                data.getRow(i, v);
                TNode *winner = getWinner(v);
                if (activation(*winner, v) >= a_t) {
                    winner->wins++;
                    //step++;
                }
            }
            step = data.rows();

            int prefMeshSize = meshNodeSet.size();
            removeLoosers();
            updateAllConnections();
            dbgOut(1) << "Finishing: " << prefMeshSize << "\t->\t" << meshNodeSet.size() << endl;

            if (maxNodeNumber == meshNodeSet.size() || meshNodeSet.size() == 1)
                break;
        } while (true);

        return *this;
    }

    /**/

    VILMAP& finishMapFixed() {

        dbgOut(1) << "Finishing map with: " << meshNodeSet.size() << endl;
        while (step != 1) { // finish the previous iteration
            trainningStep();
        }
        maxNodeNumber = meshNodeSet.size(); //fix mesh max size

        dbgOut(1) << "Finishing map with: " << meshNodeSet.size() << endl;

        trainningStep(); //step equal to 2
        while (step != 1) {
            trainningStep();
        }

        dbgOut(1) << "Finishing map with: " << meshNodeSet.size() << endl;

        return *this;
    }

    void printWinners() {

        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {

            int count = 0;
            for (int i = 0; i < data.rows(); i++) {
                TVector row;
                data.getRow(i, row);
                TNumber a = activation(*(*itMesh), row);

                if (a >= a_t) {
                    dbgOut(1) << i << " ";
                    count++;
                }
            }
            dbgOut(1) << "\t" << (*itMesh)->getId() << "\t" << (*itMesh)->wins << "\t" << count << endl;
            itMesh++;
        }
    }

    /*
    VILMAP& finishMap() {
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

        removeLoosers();
        updateAllConnections();

        maxNodeNumber = meshNodeSet.size();
        trainning(age_wins);
    }/**/

    VILMAP& resetWins() {

        //Remove os perdedores
        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {
            (*itMesh)->wins = 0;
            itMesh++;
        }

        step = 0;

        return *this;
    }

    inline SOM& trainning(int N = 1) {
        
        for (uint l = 0; l < data.rows(); l++) {
            TVector v;
            for (uint c = 0; c < data.cols(); c++) {
                if (data[l][c] != 5) {
                    v.append(data[l][c]);
                }
            }
            updateMap(v);
            
        }
        return *this;
    }

    inline void updateNodeDimension(TNode *winner1, const TVector &w) {
        //Fazer atualizacao da dimensao da rede
        //Levando em consideração 12 característcas
        int somTam = this->size();
        TVector tempVector_ds, tempVector_a, tempVector_w;
        int sizeTemp = w.size() - winner1->w.size();
        tempVector_ds.size(sizeTemp);
        tempVector_a.size(sizeTemp);
        tempVector_w.size(sizeTemp);
        tempVector_ds.fill(0.5);

        for (int i = 0, j = winner1->w.size(); i < sizeTemp; i++, j++) {
            tempVector_w[i] = w[j];
        }

        winner1->w.concat(tempVector_w); //Aumentando o vetor de prototipos preenchendo com 0
        winner1->ds.concat(tempVector_ds); //Aumentando o vetor de relevancias preenchendo com 0

        //Aumentando o vetor de distancias preenchendo com o valor maximo
        tempVector_a.fill(winner1->a.max());
        winner1->a.concat(tempVector_a);
    }

    VILMAP& updateMap(const TVector &w) {

        using namespace std;
        TNode *winner1 = 0;

        //Passo 3 : encontra o nó vencedor
        winner1 = getWinner(w); //winner
        winner1->wins++;

        //Teste de dimensão
        if (winner1->w.size() < w.size() && winner1->w.size() > 1) {
            updateNodeDimension(winner1, w);
        }

        //Passo 6: Calcula a atividade do nó vencedor

        TNumber a = activation(*winner1, w); //DS activation
        //Se a ativação obtida pelo primeiro vencedor for menor que o limiar
        //e o limite de nodos não tiver sido atingido

        if ((a < a_t) && (meshNodeSet.size() < maxNodeNumber)) {
            //dbgOut(2) << a << "\t<\t" << a_t << endl;
            //Cria um novo nodo no local do padrão observado
            TVector wNew(w);
            TNode *nodeNew = createNode(nodeID++, wNew);
            nodeNew->wins = 0; //step/meshNodeSet.size();
            nodeNew->generation = nodeNew->w.size();
            //Conecta o nodo
            updateConnections(nodeNew);

        } else if (a >= a_t) { // caso contrário
            // Atualiza o peso do vencedor
            updateNode(*winner1, w, e_b);

            //Passo 6.2: Atualiza o peso dos vizinhos
            TPNodeConnectionMap::iterator it;
            for (it = winner1->nodeMap.begin(); it != winner1->nodeMap.end(); it++) {
                TNode* node = it->first;
                updateNode(*node, w, e_n);
            }
        }
        /*
                //Passo 9:Se atingiu age_wins
                if (step >= age_wins) {

                    int size = meshNodeSet.size();
                    //remove os perdedores
                    removeLoosers();
                    dbgOut(1) << size << "\t->\t" << meshNodeSet.size() << endl;
                    //reseta o número de vitórias
                    resetWins();
                    //Passo 8.2:Adiciona conexões entre nodos semelhantes
                    updateAllConnections();
                    step = 0;
                }
         */
        step++;
        return *this;
    }

    virtual TNode *getFirstWinner(const TVector &w) {
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

    inline TNode* getWinnerCluster(const TVector &w) {
        TNode *winner = 0;
        TNumber temp = 0;

        TNumber d = dist(*(*Mesh<TNode>::meshNodeSet.begin()), w);

        winner = (*Mesh<TNode>::meshNodeSet.begin());

        TPNodeSet::iterator it;
        it = Mesh<TNode>::meshNodeSet.begin();
        it++;
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {

            temp = dist(*(*it), w);
            if ((d > temp) && ((*it)->a.size() >= w.size())) {
                d = temp;
                winner = (*it);
            }

        }


        return winner;
    }

    inline VILMAP& getWinners(const TVector &w, TNode* &winner1, TNode* &winner2) {
        TPNodeSet::iterator it = Mesh<TNode>::meshNodeSet.begin();
        TNumber minDist = dist2(*(*it), w);

        //find first winner
        winner1 = (*it);
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {
            TNumber dist = dist2(*(*it), w);
            if (dist < minDist) {
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
            if (*it != winner1) {
                TNumber dist = dist2(*(*it), w);
                if (dist < minDist) {
                    minDist = dist;
                    winner2 = (*it);
                    if (wdist(*winner1, *(*it)) <= minwd)
                        distWinner = winner2;
                }
            }
        }

        if (distWinner != NULL)
            winner2 = distWinner;

        return *this;
    }

    void getActivationVector(const TVector &sample, TVector &actVector) {
        actVector.size(Mesh<TNode>::meshNodeSet.size());

        int i = 0;
        for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end(); it++) {
            actVector[i] = activation(*(*it), sample);
            i++;
        }
    }

    bool isNoise(const TVector &w) {
        TNode *winner = getWinner(w);
        double a = activation(*winner, w);
        return (a < a_t);
    }

    void resetToDefault(int dimw = 2) {
        VILMAP::dimw = dimw;
        step = 0;
        nodesLeft = 1;

        maxNodeNumber = 100;
        e_b = 0.05;
        e_n = 0.0006;
        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;
        nodesCounter = 1;
        nodeID = 0;

        destroyMesh();
        TVector v(dimw);
        v.random();
        TVector wNew(v);
        createNode(nodeID++, wNew);
    }

    void reset(int dimw) {
        VILMAP::dimw = dimw;
        step = 0;
        nodesLeft = 1;

        counter_i = 0;
        aloc_node = 0;
        aloc_con = 0;
        nodesCounter = 1;
        nodeID = 0;

        destroyMesh();
        TVector v(dimw);
        v.random();
        TVector wNew(v);
        createNode(nodeID++, wNew);
    }

    void resetSize(int dimw) {
        VILMAP::dimw = dimw;
    }

    void binarizeRelevances() {

        TPNodeSet::iterator it;
        for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end(); it++) {
            TNode *node = *it;
            float average = node->ds.mean();
            for (int i = 0; i < node->ds.size(); i++) {
                if (node->ds[i] > average)
                    node->ds[i] = 1;
                else
                    node->ds[i] = 0;
            }
        }
    }

    void reset(void) {
        reset(dimw);
    }

    VILMAP(int dimw) {
        resetToDefault(dimw);
    };

    ~VILMAP() {
    }

    template<class Number> VILMAP& outputCentersDs(MatMatrix<Number> &m) {
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

    virtual bool saveParameters(std::ofstream &file) {

        file << maxNodeNumber << "\t";
        file << minwd << "\t";
        file << e_b << "\t";
        file << e_n << "\t";
        file << dsbeta << "\t"; //Taxa de aprendizagem
        file << epsilon_ds << "\t"; //Taxa de aprendizagem
        file << age_wins << "\t"; //period to remove nodes
        file << lp << "\t"; //remove percentage threshold
        file << a_t << "\n";
        return true;
    }

    virtual bool readParameters(std::ifstream &file) {

        file >> maxNodeNumber;
        file >> minwd;
        file >> e_b;
        file >> e_n;
        file >> dsbeta; //Taxa de aprendizagem
        file >> epsilon_ds; //Taxa de aprendizagem
        file >> age_wins; //period to remove nodes
        file >> lp; //remove percentage threshold
        file >> a_t;
        file.get(); //skip line end

        return true;
    }
};

#endif /* VILMAP_H_ */
