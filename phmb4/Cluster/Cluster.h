/*
 * File:   Cluster.h
 * Author: hans
 *
 * Created on 23 de Setembro de 2011, 10:41
 */

#ifndef CLUSTER_H
#define	CLUSTER_H
#include <vector>
#include "randomnumbers.h"
#include "DebugOut.h"
#include <cstdlib>

class Dimensions {
    std::vector<double> ranges;
    std::vector<bool> chosen;
    int free;

public:
    Dimensions(int d) {
       for (int i=0; i<d; i++) {
           ranges.push_back(0.1+(randomUniform01()*0.9));
           chosen.push_back(false);
       }
       free = d;
    }

    double range(int i) {
        return ranges[i];
    }

    void fill(double v) {
        ranges.assign(ranges.size(), v);
    }

    void setChosen(int i) {
        chosen[i]=true;
        free--;
    }

    bool wasChosen(int i) {
        return chosen[i];
    }

    int getRandNotChosen() {
        if (free==0) return -1;

        int index = rand()%free;
        for (int i=0; i<ranges.size(); i++) {
            if (!chosen[i]) {
                if (index==0) {
                    setChosen(i);
                    return i;
                }
                index--;
            }
        }
        
        return -1;
    }

    int getRandom() {
        return rand()%ranges.size();
    }
};

class ClusterDimension {

public:
    int index;
    double m;
    double sd;

    ClusterDimension(int index, double m, double sd) {
        this->index = index;
        this->m = m;
        this->sd = sd;
    }

    ClusterDimension() {}
    ClusterDimension(int index) {
        this->index = index;
    }

    void randomizeMSD(double minj, double maxj, double sdmin, double sdmax) {
        m = randomUniform(minj,maxj);
        sd = randomUniform(sdmin*maxj,sdmax*maxj);
    }

    void randomizeIndex(int N) {
        this->index = randomUniform(0,N-1);
    }
};

class Cluster {
    std::vector<ClusterDimension> clusterdimensions;
    std::vector<ClusterDimension>::iterator it;
    Dimensions *dimensions;
public:
    int n; //number of samples

    Cluster(Dimensions* dimensions) {
        this->dimensions = dimensions;
    };

    bool dimIndexExists(ClusterDimension dim) {

        for (it = clusterdimensions.begin(); it!=clusterdimensions.end(); it++)
            if (it->index == dim.index) return true;

        return false;
    }

    bool getDimension(int index, ClusterDimension& dim) {

        for (it = clusterdimensions.begin(); it!=clusterdimensions.end(); it++)
            if (it->index == index) {
                dim = *it;
                return true;
            }

        return false;
    }

    void generateRandomIndexes(int di, int d) {

        for (int i=0; i<di; i++) {
            ClusterDimension dim;
            do {
                dim.randomizeIndex(d);
            } while (dimIndexExists(dim));
            clusterdimensions.push_back(dim);
        }
    }

    bool addDimension(int index) {
        ClusterDimension dim(index);
        clusterdimensions.push_back(dim);
        return dimIndexExists(dim);
    }

    void generateRandomMSD(double minj, double sdmin, double sdmax) {
        for (it = clusterdimensions.begin(); it!=clusterdimensions.end(); it++)
            it->randomizeMSD(minj, dimensions->range(it->index), sdmin, sdmax);
    }

    double generateUniformRandomData(int index, double minj) {
        return randomUniform(minj, dimensions->range(index));
    }

    double generateGaussianRandomData(int index, double minj) {
        ClusterDimension dim;
        if (getDimension(index, dim)) {
            double data = randomGaussian(dim.m, dim.sd);
            if (data>dimensions->range(index)) data = dimensions->range(index);
            else if (data<0) data = 0;
            return data;
        }
        else {
            return randomUniform(minj, dimensions->range(index));
        }
    }

    const std::vector<ClusterDimension>& getDimensions() {
        return clusterdimensions;
    }

private:
};

#endif	/* CLUSTER_H */
