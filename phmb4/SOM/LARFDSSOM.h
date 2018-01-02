/*
 * LARFDSSOM.h
 *
 *  Created on: 2014
 *      Author: hans
 */

#ifndef LARFDSSOM_H_
#define LARFDSSOM_H_

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
    int cls;

    inline int neighbors() {
        return nodeMap.size();
    }

    GDSNodeMW(int idIn, const TVector &v) : DSNode(idIn, v), wins(0), act(0) {
    };

    ~GDSNodeMW() {
    };
};

class LARFDSSOM : public SOM<GDSNodeMW> {
public:
    uint maxNodeNumber;
    int epochs;
    float minwd;
    float e_b;
    float e_n;
    int nodesCounter;

    float push_rate;
    
    TNumber dsbeta; //Taxa de aprendizagem
    TNumber epsilon_ds; //Taxa de aprendizagem
    float age_wins;       //period to remove nodes
    float lp;           //remove percentage threshold
    float a_t;

    int nodesLeft;
    int nodeID;
    
    inline float activation(const TNode &node, const TVector &w) {

        float distance = 0;

        for (uint i = 0; i < w.size(); i++) {
            distance += node.ds[i] * qrt((w[i] - node.w[i]));
            //dbgOut(3) <<  node.a[i] << "\t" << w[i] << "\t" << node.w[i] << endl;
        }

        float sum = node.ds.sum();
        return (sum / (sum + distance + 0.0000001));

        //float r = node.ds.sum();
//        float e = (1/(qrt(node.ds.norm()) + 0.0000001));
//        float r = distance;
//        return (1 / (1 + e*r));
        
        //return exp(-qrt(e*r));
        //return (1 / (1 + qrt(e*r)));
        //return (1 / sqrt(1 + qrt(e*r)));
    }

    inline float getWinnerActivation(const TVector &w) {
        TNode* winner = getWinner(w);        
        float a = activation(*winner, w);
        dbgOut(2) << winner->getId() << "\t" << a << endl;
        return a;
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
        
        TVector newA(node.a);
        
        //update averages
        for (uint i = 0; i < node.a.size(); i++) {
            //update neuron weights
            float distance = fabs(w[i] - node.w[i]);
            newA[i] = e*dsbeta* distance + (1 - e*dsbeta) * newA[i];
        }

        float max = newA.max();
        float min = newA.min();
        float average = newA.mean();
        //float dsa = node.ds.mean();


        //update neuron ds weights
        for (uint i = 0; i < newA.size(); i++) {
            if ((max - min) != 0) {
                //node.ds[i] = 1 - (node.a[i] - min) / (max - min);
                node.ds[i] = 1/(1+exp((newA[i]-average)/((max - min)*epsilon_ds)));
            }
            else
                node.ds[i] = 1;

//            if (node.ds[i] < 0.95*dsa)
//                node.ds[i] = epsilon_ds;

//            if (node.ds[i] < epsilon_ds)
//                node.ds[i] = epsilon_ds;
            
            node.a[i] = newA[i];
        }

        //Passo 6.1: Atualiza o peso do vencedor
        //Atualiza o nó vencedor
        node.w = node.w + e * (w - node.w);
    }

    LARFDSSOM& updateConnections(TNode *node) {
        
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
    
    LARFDSSOM& updateAllConnections() {

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

    LARFDSSOM& removeLoosers() {

//        enumerateNodes();

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

        //printWinners();
        return *this;
    }

    //*
    LARFDSSOM& finishMap(bool sorted, std::vector<int> groups) {

        dbgOut(1) << "Finishing map..." << endl;
        do {
            resetWins();
            maxNodeNumber = meshNodeSet.size();
            
            if (!sorted) {
                trainning(age_wins, groups);
            } else {
                orderedTrainning(age_wins, groups);
            }
            
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

            int prefMeshSize = meshNodeSet.size();
            removeLoosers();
            updateAllConnections();
            dbgOut(1) << "Finishing: " << prefMeshSize << "\t->\t" << meshNodeSet.size() << endl;

            if (maxNodeNumber == meshNodeSet.size() || meshNodeSet.size()==1)
                break;
        } while (true);
        
        return *this;
    }
    /**/
    
    void runTrainingStep(bool sorted, std::vector<int> groups) {
        if (sorted) {
            trainningStep(step%data.rows(), groups);
        } else {
            trainningStep(rand()%data.rows(), groups);
        }
    }
    
    LARFDSSOM& finishMapFixed(bool sorted, std::vector<int> groups) {

//        dbgOut(1) << "Finishing map with: " << meshNodeSet.size() << endl;
        while (step!=1) { // finish the previous iteration
            runTrainingStep(sorted, groups);
        }
        maxNodeNumber = meshNodeSet.size(); //fix mesh max size
        
//        dbgOut(1) << "Finishing map with: " << meshNodeSet.size() << endl;
        
        //step equal to 2
        runTrainingStep(sorted, groups);
        
        while (step!=1) {
            runTrainingStep(sorted, groups);
        }
        
//        dbgOut(1) << "Finishing map with: " << meshNodeSet.size() << endl;
        
        return *this;
    }
    
    void checkNodeClasses (const std::string &filename) {
        TPNodeSet::iterator it;
        it = Mesh<TNode>::meshNodeSet.begin();
        it++;
        int count = 0;
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {
            if ((*it)->cls == noCls) {
                count++;
            }
        }
        
        if (count > 0) {
            std::ofstream file;
            file.open(filename.c_str());

            if (!file.is_open()) {
                dbgOut(0) << "Error openning output file" << endl;
                return;
            }

            file << "Finishing with " << count << " of " << meshNodeSet.size() << " nodes unknown (" << (float)count / meshNodeSet.size() << " %)" << endl;
            dbgOut(1) << "Finishing with " << count << " of " << meshNodeSet.size() << " nodes unknown (" << (float)count / meshNodeSet.size() << " %)" <<  endl;
            
            file.close();
        }
    }
    
    void printWinners() {

        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {
            
            int count = 0;
            for (int i=0; i<data.rows(); i++) {
                TVector row;
                data.getRow(i, row);
                TNumber a = activation(*(*itMesh), row);
                
                if (a>=a_t) {
                        dbgOut(1) << i << " ";
                        count++;
                }
            }
            dbgOut(1) << "\t" << (*itMesh)->getId() << "\t" << (*itMesh)->wins << "\t" << count << endl;
            itMesh++;
        }
    }

    /*
    LARFDSSOM& finishMap() {
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

    LARFDSSOM& resetWins() {

        //Remove os perdedores
        TPNodeSet::iterator itMesh = meshNodeSet.begin();
        while (itMesh != meshNodeSet.end()) {
             (*itMesh)->wins = 0;
             itMesh++;
        }

        step = 0;

        return *this;
    }

    void createNodeMap (const TVector& w, int cls) {
        // cria um novo nodo na posição da amostra
        TVector wNew(w);
        TNode *nodeNew = createNode(nodeID, wNew);
        nodeID++;
        nodeNew->cls = cls;
        nodeNew->wins = 0;

        updateConnections(nodeNew);
    }
    
    void ageWinsCriterion(){
        //Passo 9:Se atingiu age_wins
        if (step >= age_wins) {

            int size = meshNodeSet.size();
            //remove os perdedores
            removeLoosers();
//            dbgOut(1) << size << "\t->\t" << meshNodeSet.size() << endl;
            //reseta o número de vitórias
            resetWins();
            //Passo 8.2:Adiciona conexões entre nodos semelhantes
            updateAllConnections();
            step = 0;
        }
    }
    
    LARFDSSOM& updateMap(const TVector &w) {

        using namespace std;
        TNode *winner1 = 0;

        //Passo 3 : encontra o nó vencedor
        winner1 = getWinner(w); //winner
        
        if (winner1 == NULL) {
            createNodeMap(w, noCls);
            
        } else {

            //Passo 6: Calcula a atividade do nó vencedor
            TNumber a = activation(*winner1, w); //DS activation
            //Se a ativação obtida pelo primeiro vencedor for menor que o limiar
            //e o limite de nodos não tiver sido atingido

            if ((a < a_t) && (meshNodeSet.size() < maxNodeNumber)) {
                createNodeMap(w, noCls);
                
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
        }

        ageWinsCriterion();
        
        step++;
        
        return *this;
    }
    
    LARFDSSOM& updateMapSup(const TVector& w, int cls) {
        using namespace std;
        
        TNode *winner1 = 0;
        winner1 = getFirstWinner(w); // encontra o nó vencedor

        if (winner1 == NULL) { // mapa vazio, primeira amostra
            createNodeMap(w, cls);
            
        } else if (winner1->cls != noCls && winner1->cls != cls) { // winner tem classe diferente da amostra
            // caso winner seja de classe diferente, checar se existe algum
            // outro nodo no mapa que esteja no raio a_t da nova amostra e
            // que pertença a mesma classe da mesma
            
            handleDifferentClass(winner1, w, cls);
            
        } else { // winner1 representativo e da mesma classe da amostra 
            
            if ((winner1->act < a_t) && (meshNodeSet.size() < maxNodeNumber)) {
                // cria um novo nodo na posição da amostra
                createNodeMap(w, cls);

            } else if (winner1->act >= a_t) {
                winner1->wins++;
                winner1->cls = cls;
                updateNode(*winner1, w, e_b);

                TPNodeConnectionMap::iterator it;
                for (it = winner1->nodeMap.begin(); it != winner1->nodeMap.end(); it++) {            
                    TNode* node = it->first;
                    updateNode(*node, w, e_n);
                }
            } 
        }

        ageWinsCriterion();
        
        step++;
        
        return *this;
    }
    
    void handleDifferentClass(TNode *winner1, const TVector& w, int cls) {
        TNode *newWinner = winner1;
        while((newWinner = getNextWinner(newWinner)) != NULL) { // saiu do raio da ativação -> não há um novo vencedor
            if (newWinner->cls == noCls || newWinner->cls == cls) { // novo vencedor valido encontrado
                break;
            }
        }

        if (newWinner != NULL) { // novo winner de acordo com o raio de a_t
            newWinner->cls = cls;
            newWinner->wins++;
            // puxar o novo vencedor
            updateNode(*newWinner, w, e_b);

            TPNodeConnectionMap::iterator it;
            for (it = newWinner->nodeMap.begin(); it != newWinner->nodeMap.end(); it++) {            
                TNode* node = it->first;
                updateNode(*node, w, e_n);
            }

            // empurrar o primeiro winner que tem classe diferente da amostra
            updateNode(*winner1, w, -push_rate);

        } else if (meshNodeSet.size() < maxNodeNumber) {

            // cria um novo nodo na posição da amostra
            createNodeMap(w, cls);

//            TVector wNew(winner1->w);
//            TNode *nodeNew = createNode(nodeID++, wNew);
//            nodeNew->cls = cls;
//            nodeNew->wins = 0;

//                TVector aNew(winner1->a);
//                nodeNew->a = aNew;
//                TVector dsNew(winner1->ds);
//                nodeNew->ds = dsNew;
//                
//                nodeNew->act = winner1->act;
//                nodeNew->step = winner1->step;
//                nodeNew->touched = winner1->touched;
//                nodeNew->wins = winner1->wins + 1;               
            // puxar o vencedor
//            updateNode(*nodeNew, w, e_b);

//            updateNode(*winner1, w, -push_rate);
        }
    }

    virtual TNode *getFirstWinner(const TVector &w){
        TNode *winner = 0;
        TNumber temp = 0;
        
        if (meshNodeSet.empty()) {
            return NULL;
        }
        
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

    inline int getWinnerClass(const TVector &w) {
        TNode *winner = getWinner(w);

        return winner->cls;
    }
    
    inline TNode* getWinner(const TVector &w) {
        TNode *winner = 0;
        TNumber temp = 0;
        
        if (meshNodeSet.empty()) {
            return NULL;
        }
        
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
    
    void getActivationVector(const TVector &sample, TVector &actVector) {
        actVector.size(Mesh<TNode>::meshNodeSet.size());
        
        int i=0;
        for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end(); it++) {
            actVector[i] = activation(*(*it), sample);
            i++;
        }
    }
    
    bool isNoise(const TVector &w) {
        TNode *winner = getWinner(w);
        double a = activation(*winner, w);
        return (a<a_t);
    }

    void resetToDefault(int dimw = 2) {
        LARFDSSOM::dimw = dimw;
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
        createNode(0, wNew);
    }

    void reset(int dimw) {
        LARFDSSOM::dimw = dimw;
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
        TNode *nodeNew = createNode(0, wNew);
        nodeNew->cls = noCls;
    }
    
    void resetSize(int dimw) {
        LARFDSSOM::dimw = dimw;
    }
    
    void binarizeRelevances() {
        
        TPNodeSet::iterator it;
        for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end(); it++) {
            TNode *node = *it;
            float average = node->ds.mean();
            for (int i=0; i<node->ds.size(); i++) {
                if (node->ds[i]>average)
                    node->ds[i] = 1;
                else
                    node->ds[i] = 0;
            }
        }
    }

    void reset(void) {
        reset(dimw);
    }

    LARFDSSOM(int dimw) {
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
    
    virtual bool saveParameters(std::ofstream &file) {
        
        file << maxNodeNumber << "\t";
        file << minwd << "\t";
        file << e_b << "\t";
        file << e_n << "\t";
        file << dsbeta << "\t"; //Taxa de aprendizagem
        file << epsilon_ds << "\t"; //Taxa de aprendizagem
        file << age_wins << "\t";       //period to remove nodes
        file << lp << "\t";          //remove percentage threshold
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
        file >> age_wins;       //period to remove nodes
        file >> lp;           //remove percentage threshold
        file >> a_t;
        file.get();//skip line end
        
        return true;
    }
};

#endif /* LARFDSSOM_H_ */