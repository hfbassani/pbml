/*
 * SOM.h
 *
 *  Created on: 10/02/2009
 *      Author: daniel
 */

#ifndef SOM_H_
#define SOM_H_

#include <list>
#include <set>
#include <chrono>
#include <random>
#include "Mesh.h"
#include "MatVector.h"


template<class T> class SOM :public Mesh<T>{
public:

    typedef T TNode;
    typedef typename TNode::TNumber TNumber;
    typedef typename TNode::TVector TVector;
    typedef typename TNode::TConnection TConnection;
    typedef typename TNode::TPNodeConnectionMap TPNodeConnectionMap;
    typedef std::set<TNode*> TPNodeSet; // mapeamento global dos n�s
    typedef std::set<TConnection*> TPConnectionSet; //Para mapeamento global das conex�es da mesh
    typedef MatMatrix<TNumber> TMatrix;

    TMatrix data;
    int step;
    int counter_i;
    int dimw;//dimensão do vetor de pessos
    
    float supervisionRate;
    float reinforcementRate;
    float unsupervisionRate;
    
    int noCls;
//    MatVector<int> reinforcementIndexes;
//    MatVector<int> supervisedIndexes;
    
    virtual inline int getWinnerClass(const TVector &w) {
        return noCls;
    }
    
    virtual inline TNode* getWinner(const TVector &w) {
        TNode *winner = 0;
        TNumber temp = 0;
        TNumber d = (*Mesh<TNode>::meshNodeSet.begin())->w.dist(w);
        winner = (*Mesh<TNode>::meshNodeSet.begin());

        typename TPNodeSet::iterator it;
        it = Mesh<TNode>::meshNodeSet.begin();
        it++;
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {
            temp = (*it)->w.dist(w);
            if (d > temp) {
                d = temp;
                winner = (*it);
            }
        }

        return winner;
    }

    virtual inline SOM& getWinners(const TVector &w, TNode* &winner1, TNode* &winner2) {
        typename TPNodeSet::iterator it = Mesh<TNode>::meshNodeSet.begin();
        TNumber temp = 0;
        TNumber d2, d1 = (*it)->w.dist2(w);
        winner1 = (*it);
        it++;
        if ((*it)->w.dist2(w) < d1) {
            winner2 = winner1;
            d2 = d1;
            winner1 = (*it);
            d1 = (*it)->w.dist2(w);
        } else {
            winner2 = (*it);
            d2 = (*it)->w.dist2(w);
        }
        it++;
        for (; it != Mesh<TNode>::meshNodeSet.end(); it++) {
            temp = (*it)->w.dist2(w);
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

    virtual inline SOM& getWinners(const TVector &w, TVector &w1, TVector &w2) {
        TNode *winner1, *winner2;
        getWinners(w,winner1,winner2);
        w1=winner1->w;
        w2=winner2->w;
        return *this;
    }

    virtual SOM& updateMap(const TVector &w) = 0;
    
    virtual SOM& updateMapSup(const TVector &w, int cls) = 0;

    SOM& trainningEach(int N = 1) {
        MatVector<int> vindex(data.rows());
        vindex.range(0, vindex.size() - 1);
        
        int vSize=vindex.size();
        vindex.srandom(MatUtils::getCurrentCPUTimer());
        vindex.shuffler();
        
        TVector v(data.cols());
        for (int n = 0; n < N; n++) {
            for (uint l = 0; l < data.cols(); l++)
                v[l] = data[vindex[n%vSize]][l];
            updateMap(v);
        }
        return *this;
    }
    
    SOM& trainning(int epochs = 1, std::vector<int> groups = NULL) {
//        trainningStep(rand()%data.rows(), groups);
        
        for (int epoch = 0 ; epoch < epochs ; epoch++)
            for (int row = 0 ; row < data.rows() ; ++row)
                trainningStep(rand()%data.rows(), groups);
                
        return *this;
    }
    
    SOM& orderedTrainning(int epochs = 1, std::vector<int> groups = NULL) {
        for (int epoch = 0 ; epoch < epochs ; epoch++)
            for (int row = 0 ; row < data.rows() ; ++row)
                trainningStep(row, groups);
        
        return *this;
    }
   
    SOM& trainningStep(int row = 1, std::vector<int> groups = NULL) {
        TVector v(data.cols());
        for (uint l = 0; l < data.cols(); l++)
                v[l] = data[row][l];
        
        chooseTrainingType(v, groups[row]);
        
        return *this;
    }
    
    SOM& trainningStep(TVector &v) {

        updateMap(v);
        
        return *this;
    }
    
    void chooseTrainingType(TVector &v, int cls) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        std::uniform_real_distribution<double> distribution (0.0,1.0);
        double rate = distribution(generator);
        
        if (rate <= supervisionRate) { //supervised
//            dbgOut(1) << "supervised" << endl << endl;
            updateMapSup(v, cls);
        } else { //unsupervised
//            dbgOut(1) << "unsupervised" << endl << endl;
            updateMap(v); 
        }
    }

    SOM()
    {
        counter_i=0;
    }
    
    typename TPNodeSet::iterator it;

    TNode *getFirstNode() {
        it = Mesh<TNode>::meshNodeSet.begin();
        return (*it);
    }

    TNode *getNextNode() {
        it++;
        return (*it);
    }

    bool finished() {
        return (it == Mesh<TNode>::meshNodeSet.end());
    }    
    
    virtual bool isNoise(const TVector &w) {
        return false;
    }
    
    virtual TNode *getFirstWinner(const TVector &w){
        return getWinner(w);
    }
    
    virtual TNode *getNextWinner(TNode *previowsWinner) {
        return NULL;
    }
    
    virtual bool saveParameters(std::ofstream &file) {
        
        return true;
    }
    
    virtual bool saveSOM(const std::string &filename) {
        
        std::ofstream file(filename.c_str());
        file.precision(16);
        
        if (!file.is_open()) {
            return false;
        }
        
        saveParameters(file);
        
        for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end(); it++) {
            (*it)->write(file);
        }
        
        file.close();
        return true;
    }
    
    virtual int getInputSize() {
        it = Mesh<TNode>::meshNodeSet.begin();
        if (it == Mesh<TNode>::meshNodeSet.end()) return 0;
        else return (*it)->w.size();
    }
    
    virtual bool readParameters(std::ifstream &file) {
        
        return true;
    }
    
    virtual bool readSOM(const std::string &filename) {
        
        std::ifstream file(filename.c_str());
        
        if (!file.is_open()) {
            return false;
        }
        
        readParameters(file);
        
        int id = 0;
        Mesh<TNode>::destroyMesh();
        while (!file.eof()) {
            TVector v(1);
            TNode* node = new TNode(id, v);
            node->read(file);
            if (file.eof()) {
                delete node;
                break;
            }
            Mesh<TNode>::meshNodeSet.insert(node);
            id++;
        }
        
        return true;
    }
    
    virtual void printMesh() {
        std::cout << Mesh<TNode>::meshNodeSet.size() << std::endl;
        int i=1;
        for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end(); it++) {
             std::cout << i << ":" << (*it)->w.toString() << std::endl;
             i++;
        }
    }
};

template<class T> inline double sumOfSquaredError(SOM<T> &som) {
    return sumOfSquaredError(som.data,som);
};

#endif /* SOM_H_ */