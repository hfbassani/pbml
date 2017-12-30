/*
 * LVQ.h
 *
 *  Created on: 10/02/2009
 *      Author: daniel
 */

#ifndef LVQ_H_
#define LVQ_H_

#include <list>
#include <set>
#include "Mesh.h"
#include "MatVector.h"
#include <limits>

template<class T> class LVQ :public Mesh<T>{
public:

    typedef T TNode;
    typedef typename TNode::TNumber TNumber;
    typedef typename TNode::TVector TVector;
    typedef std::set<TNode*> TPNodeSet; // mapeamento global dos n�s
    typedef MatMatrix<TNumber> TMatrix;
    typedef std::vector<int> TClassVector;

    TMatrix data;
    TClassVector vcls;
    int step;
    int counter_i;
    int dimw; //dimensão do vetor de pesos
    
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

    virtual inline LVQ& getWinners(const TVector &w, TNode* &winner1, TNode* &winner2) {
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
    
    virtual inline LVQ& getWinnerRunnerUp(const TVector &w, TNode* &winner, TNode* &runnerUp, int cls) {
        TNumber dw = std::numeric_limits<TNumber>::max();
        TNumber dr = std::numeric_limits<TNumber>::max();
        
        winner = NULL;
        runnerUp = NULL;
        
        typename TPNodeSet::iterator it;
        for (it = Mesh<TNode>::meshNodeSet.begin(); it != Mesh<TNode>::meshNodeSet.end(); it++) {
            if ((*it)->cls == cls) {
                TNumber tempW = (*it)->w.dist(w);
                if (dw > tempW) {
                    dw = tempW;
                    winner = (*it);
                }
            }
            else {
                TNumber tempR = (*it)->w.dist(w);
                if (dr > tempR) {
                    dr = tempR;
                    runnerUp = (*it);
                }                
            }
        }

        return *this;
    }

    virtual inline LVQ& getWinners(const TVector &w, TVector &w1, TVector &w2) {
        TNode *winner1, *winner2;
        getWinners(w,winner1,winner2);
        w1=winner1->w;
        w2=winner2->w;
        return *this;
    }

    virtual LVQ& updateMap(const TVector &w, int cls) = 0;

    LVQ& trainningEach2(int N = 1) {
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
    
    LVQ& trainningEach(int N = 1) {
        MatVector<int> vindex(data.rows());
        vindex.range(0, vindex.size() - 1);
        TNumber total_change = 0;
        
        int vSize=vindex.size();
        vindex.srandom(MatUtils::getCurrentCPUTimer());
        vindex.shuffler();
        
        TVector v;
        for (int n = 0; n < N; n++) {
            for (int l = 0; l < vindex.size(); l++) {
                data.getRow(vindex[l], v);
                total_change = updateMap(v, vcls[vindex[l]]);
            }
        }
        return *this;
    }
    
    LVQ& trainning(int N = 1) {
        for (int i=0; i<N; i++)
            trainningStep();
        
        return *this;
    }
    
    virtual LVQ& trainningStep() {
        TVector v;
        
        if (data.rows()>0) {
            int row = rand()%data.rows();
            //std::cout << "Row: " << row << std::endl;
            data.getRow(row, v);
            updateMap(v, vcls[row]);        
        }
        return *this;
    }
    
    
    
    virtual LVQ& trainningStep(TVector &v, int cls) {

        updateMap(v, cls);        
        return *this;
    }
    

    LVQ()
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
    
    virtual bool saveLVQ(const std::string &filename) {
        
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
    
    virtual bool readLVQBD(const std::string &filename) {
        std::ifstream file(filename.c_str());
        float d;
        std::string line = "";
        
        if (!file.is_open()) {
            return false;
        }
        
        readParameters(file); //Le primeira linha dos paramentos
            
        int posicaoAtual = file.tellg(); //guarda posicao atual do arquivo
        getline(file, line, '\n'); //le linha para calcular o numero de colunas
        std::stringstream parser(line);
        
        int col=0;      
        while (true) {
              parser >> d;
              if (parser.fail()) break;
              if (parser.eof()) break;
              if (parser.bad()) break;                      
              col++;
        }
        file.seekg(posicaoAtual); //Retornar para a segunda linha do arquivo

        int id = 0;
        Mesh<TNode>::destroyMesh();
        while (!file.eof()) {
            TVector v(col);
            TNode* node = new TNode(id, v);
            node->read(file);
            if (file.eof()) {
                delete node;
                break;
            }
            Mesh<TNode>::meshNodeSet.insert(Mesh<TNode>::meshNodeSet.end(), node);
            id++;
        }
        
        return true;
    }
    
    virtual bool readLVQ(const std::string &filename) {
        
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
            Mesh<TNode>::meshNodeSet.insert(Mesh<TNode>::meshNodeSet.end(), node);
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

template<class T> inline double sumOfSquaredError(LVQ<T> &som) {
    return sumOfSquaredError(som.data,som);
};

#endif /* LVQ_H_ */
