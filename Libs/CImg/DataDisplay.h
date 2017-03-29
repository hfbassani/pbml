/* 
 * File:   DataDisplay.h
 * Author: hans
 *
 * Created on 29 de Março de 2012, 08:56
 */

#ifndef DATADISPLAY_H
#define	DATADISPLAY_H

#include <sstream>
#include <time.h>
#include "CImg.h"
#include "SOM.h"
#include "DSNode.h"
#include "SOM2D.h"
#include "DSSOM.h"
#include "DSSOMC.h"
#include "DSNeuron.h"
#include "DebugOut.h"

using namespace cimg_library;

const unsigned char contour[] = {0, 0, 0};
const unsigned char background[] = {255, 255, 255};

#define KEY_SLEEP 100
#define HUE_START 50
#define MAX_HUE 300

class DataDisplay {
private:

    CImg<unsigned char> *image;
    CImgDisplay *disp;
    MatMatrix<float> *trainingData;
    std::vector<std::vector<int> > trueClusters;
    
    int X;
    int Y;
    bool drawNodes;
    bool bmucolor;
    bool trueClustersColor;
    bool drawConnections;
    bool filterNoise;
    bool pause;
    int padding;
    int gitter;
    MatMatrix<float> *averages;
    map<int, int> *groupLabels;
    int maxDataPlot;
    
    clock_t now;
    clock_t last;

public:

    DataDisplay(MatMatrix<float> *trainingData, MatMatrix<float> *averages = NULL, map<int, int> *groupLabels = NULL, int padding = 20, int gitter = 0, bool bmucolor = true, bool trueClustersColor = true, bool filterNoise = false) {

        this->trainingData = trainingData;
        setAverages(averages);
        setGroupLabels(groupLabels);
        setBmucolor(bmucolor);
        setPadding(padding);
        setGitter(gitter);
        setTrueClustersColor(trueClustersColor);
        setDrawNodes(true);
        setDrawConnections(true);
        setFilterNoise(filterNoise);
        pause = false;
        now = last = 0;
        maxDataPlot = -1;

        image = new CImg<unsigned char>(500, 500, 1, 3);
        disp = new CImgDisplay(500, 500);

        X = 0;
        Y = 1;
    }

    void setGroupLabels(map<int, int>* groupLabels) {
        this->groupLabels = groupLabels;
    }

    void setAverages(MatMatrix<float>* averages) {
        this->averages = averages;
    }

    void setDrawNodes(bool drawNodes) {
        this->drawNodes = drawNodes;
    }
    
    void setBmucolor(bool bmucolor) {
        this->bmucolor = bmucolor;
    }

    void setTrueClustersColor(bool trueClustersColor) {
        this->trueClustersColor = trueClustersColor;
    }
    
    void setFilterNoise(bool filterNoise) {
        this->filterNoise = filterNoise;
    }

    void setGitter(int gitter) {
        this->gitter = gitter;
    }

    void setPadding(int padding) {
        this->padding = padding;
    }
    
    void setMaxDataPlot(int max) {
        this->maxDataPlot = max;
    }

    void setY(int Y) {
        this->Y = Y;
    }

    void setX(int X) {
        this->X = X;
    }

    void setDrawConnections(bool drawConnections) {
        this->drawConnections = drawConnections;
    }

    bool isDrawConnections() const {
        return drawConnections;
    }

    ~DataDisplay() {
        delete image;
        delete disp;
    }

    void updateTitle() {
        stringstream str;
        str << "View:" << X << "X" << Y;
        if (trueClustersColor) 
            str << " - True Clusters Colors";
        if (filterNoise)
            str << " - Filter Noise";
        
        disp->set_title(str.str().c_str());
    }
    
    template <class SOMType>
    void display(SOMType &som, MatVector<float> *data = NULL) {

        processKey();
        updateTitle();
        if (!pause && !disp->is_closed()) {
            buildImage(som, X, Y, data);
            image->display(*disp);
        }
    }

    template <class SOMType>
    void display(SOMType &som, int X, int Y, MatVector<float> *data = NULL) {
        updateTitle();

        if (!pause && !disp->is_closed()) {
            buildImage(som, X, Y, data);
            image->display(*disp);
        }
    }

    template <class SOMType>
    void displayLoop(SOMType &som, MatVector<float> *data = NULL) {

        while (!disp->is_closed()) {
            processKey();

            updateTitle();

            if (!pause) {
                buildImage(som, X, Y, data);
                image->display(*disp);
            }
            //disp->wait();
        }
    }

    void close() {
        disp->close();
    }

    void setTrueClustersData(std::vector<std::vector<int> > &trueClusters) {
        this->trueClusters = trueClusters;
    }
    
    void setTrueClustersData(std::vector<int> &groups) {
        trueClusters.clear();
        std::map<int, std::vector<int> > gm;
        
        for (int i=0; i<groups.size(); i++) {
            gm[groups[i]].push_back(i);
            //dbgOut(0) << "cluster " << groups[i] << " size: " << gm[groups[i]].size() << endl; 
        }
        
        for (int i=0; i<gm.size(); i++) {
            trueClusters.push_back(gm[i]);
        }
    }
    
    void loadTrueClustersData(std::string filename) {

        trueClusters.clear();
        string line;

        //Read true clusters
        int period = filename.find_last_of(".");
        string trueClustersFileName = filename.substr(0, period) + ".true";
        ifstream trueFile(trueClustersFileName.c_str());

        if (!trueFile.is_open()) {
            dbgOut(1) << "Error reading true clusters file:" << trueClustersFileName << endl;
            return;
        }

        trueFile.ignore(4); //Skip "DIM="
        int dim;
        trueFile >> dim; //Read number of dimensions
        getline(trueFile, line); //Skip first line

        int dimvalues;
        int clusterSize;
        int objectIndex;
        int clust = 0;
        while (true) {
            for (int i = 0; i < dim; i++) {//Skip dimension values
                trueFile >> dimvalues;
                if (trueFile.eof())
                    break;
            }
            if (trueFile.eof()) break;

            trueFile >> clusterSize;

            std::vector<int> cluster;
            for (int i = 0; i < clusterSize; i++) {
                trueFile >> objectIndex;
                if (trueFile.eof()) break;
                cluster.push_back(objectIndex);
            }
            trueClusters.push_back(cluster);
            clust++;
        }
        trueFile.close();
    }

    void setPause(bool pause) {
        this->pause = pause;
    }

    bool isPaused() const {
        return pause;
    }

private:

    void processKey() {
        now = clock() / (CLOCKS_PER_SEC / 1000);
        
        if (now - last>KEY_SLEEP) {
            
            if (disp->key() == cimg::keyP) {
                pause = !pause;
            }
                        
            if (disp->key() == cimg::keyARROWUP) {
                Y++;
                if (Y >= trainingData->cols())
                    Y = 0;
            }

            if (disp->key() == cimg::keyARROWDOWN) {
                Y--;
                if (Y < 0)
                    Y = trainingData->cols() - 1;
            }

            if (disp->key() == cimg::keyARROWRIGHT) {
                X++;
                if (X >= trainingData->cols())
                    X = 0;
            }

            if (disp->key() == cimg::keyARROWLEFT) {
                X--;
                if (X < 0)
                    X = trainingData->cols() - 1;
            }

            if (disp->key() == cimg::keyESC) {
                disp->close();
            }

            if (disp->key() == cimg::keyU) {
                drawNodes = !drawNodes;
            }
            
            if (disp->key() == cimg::keyC) {
                drawConnections = !drawConnections;
            }

            if (disp->key() == cimg::keyT) {
                trueClustersColor = !trueClustersColor;
            }

            if (disp->key() == cimg::keyN) {
                filterNoise = !filterNoise;
            }

            if (disp->key() == cimg::keyB) {
                bmucolor = !bmucolor;
            }            

            if (disp->key() == cimg::keyPADADD) {
                gitter+=5;
            }

            if (disp->key() == cimg::keyPADSUB) {
                if (gitter>0)
                    gitter-=5;   
            }

            last = now;
        }
    }

    void buildImage(SOM<DSNode> &som, int X, int Y, MatVector<float> *dataVector, bool clean = true) {

        som.enumerateNodes();
        unsigned char bmuColor[3];
        int width = (image->width() - 2 * padding);
        int height = (image->height() - 2 * padding);

        if (clean) image->fill(background[0], background[1], background[2]);

        //Draw data
        if (trueClustersColor && trueClusters.size() > 0)
            drawTrueClusters(X, Y);
        else{
            int size = trainingData->rows()-1;
            if (maxDataPlot>0)
                size = maxDataPlot;
                
            for (int k = 0; k < size; k++) {
                int i = k;
                if (maxDataPlot>0)
                        i = rand()%size;

                MatVector<float> row;
                trainingData->getRow(i, row);
                if (filterNoise && som.isNoise(row))
                    continue;
                
                int gitterx = applyGitter(gitter);
                int gittery = applyGitter(gitter);

                if (bmucolor) {
                    
                    DSNode *bmu = som.getWinner(row);

                    int r, g, b;
                    int size = som.size()-1;
                    if (size==0) size = 1;
                    int h = HUE_START + bmu->getId()*MAX_HUE / (size);
                    HSVtoRGB(&r, &g, &b, h, 255, 255);
                    bmuColor[0] = r;
                    bmuColor[1] = g;
                    bmuColor[2] = b;

                    image->draw_circle(padding + (*trainingData)[i][X] * width + gitterx, padding + (*trainingData)[i][Y] * height + gittery, 2, bmuColor);
                }
                else
                     image->draw_circle(padding + (*trainingData)[i][X] * width + gitterx, padding + (*trainingData)[i][Y] * height + gittery, 2, contour);
            }
        }

        //plot dataVector
        if (dataVector != NULL) {
            DSNode *bmu = som.getWinner(*dataVector);

            int nx = bmu->w[X] * image->width();
            int ny = bmu->w[Y] * image->height();
            int dx = (*dataVector)[X] * image->width();
            int dy = (*dataVector)[Y] * image->height();

            int r, g, b;
            int size = som.size()-1;
            if (size==0) size = 1;
            int h = HUE_START + bmu->getId()*MAX_HUE / (size);
            HSVtoRGB(&r, &g, &b, h, 255, 255);
            bmuColor[0] = r;
            bmuColor[1] = g;
            bmuColor[2] = b;

            image->draw_circle(padding + (*dataVector)[X] * width, padding + (*dataVector)[Y] * height, 3, bmuColor);
            image->draw_line(padding + nx, padding + ny, padding + dx, padding + dy, bmuColor);
        }

        //Draw centers
        if (drawNodes) {
            MatMatrix<float> centers;
            som.outputCenters(centers);
            for (DSNode *bmu = som.getFirstNode(); !som.finished(); bmu = som.getNextNode()) {
                int r, g, b;
                int size = som.size()-1;
                if (size==0) size = 1;
                int h = HUE_START + bmu->getId()*MAX_HUE / (size);
                HSVtoRGB(&r, &g, &b, h, 255, 255);
                bmuColor[0] = r;
                bmuColor[1] = g;
                bmuColor[2] = b;

                image->draw_circle(padding + bmu->w[X] * width, padding + bmu->w[Y] * height, 4, contour);
                image->draw_circle(padding + bmu->w[X] * width, padding + bmu->w[Y] * height, 3, bmuColor);
                int cx = bmu->w[X] * width;
                int cy = bmu->w[Y] * height;
                int x0 = bmu->w[X] * width - bmu->ds[X]*20;
                int x1 = bmu->w[X] * width + bmu->ds[X]*20;
                int y0 = bmu->w[Y] * height - bmu->ds[Y]*20;
                int y1 = bmu->w[Y] * height + bmu->ds[Y]*20;
                image->draw_line(padding + cx, padding + y0, padding + cx, padding + y1, contour);
                image->draw_line(padding + x0, padding + cy, padding + x1, padding + cy, contour);
            }

            //draw labels
            if (averages != NULL) {
                for (int labelIndex = 0; labelIndex < averages->cols(); labelIndex++) {
                    std::stringstream sstr;
                    sstr << (*groupLabels)[labelIndex];
                    float x = averages->get(X, labelIndex);
                    float y = averages->get(Y, labelIndex);
                    image->draw_text(padding + x * width, padding + y * height, sstr.str().c_str(), contour);
                }
            }
        }

        //Draw connections
        if (drawConnections) {
            SOM<DSNode>::TPConnectionSet::iterator it;
            for (it = som.meshConnectionSet.begin(); it != som.meshConnectionSet.end(); it++) {
                float x0 = (*it)->node[0]->w[X];
                float y0 = (*it)->node[0]->w[Y];
                float x1 = (*it)->node[1]->w[X];
                float y1 = (*it)->node[1]->w[Y];
                float dist = (*it)->node[0]->ds.dist((*it)->node[1]->ds) / sqrt((*it)->node[1]->ds.size());
                unsigned char conColor[] = {255, 255, 255};
                conColor[0] = conColor[0] * dist;
                conColor[1] = conColor[1] * dist;
                conColor[2] = conColor[2] * dist;
                image->draw_line(padding + x0 * width, padding + y0 * height, padding + x1 * width, padding + y1 * height, conColor);
            }
        }
    }

    void buildImage(SOM<NodeW> &som, int X, int Y, MatVector<float> *dataVector, bool clean = true) {

        unsigned char bmuColor[3];
        int width = (image->width() - 2 * padding);
        int height = (image->height() - 2 * padding);

        if (clean) image->fill(background[0], background[1], background[2]);

        //Draw data
        if (trueClustersColor && trueClusters.size() > 0)
            drawTrueClusters(X, Y);
        else
            for (int i = 0; i < trainingData->rows(); i++) {

                int gitterx = applyGitter(gitter);
                int gittery = applyGitter(gitter);

                if (bmucolor) {
                    MatVector<float> row;
                    trainingData->getRow(i, row);
                    NodeW *bmu = som.getWinner(row);

                    int r, g, b;
                    int size = som.size()-1;
                    if (size==0) size = 1;
                    int h = HUE_START + bmu->getId()*MAX_HUE / (size);
                    HSVtoRGB(&r, &g, &b, h, 255, 255);
                    bmuColor[0] = r;
                    bmuColor[1] = g;
                    bmuColor[2] = b;

                    image->draw_circle(padding + (*trainingData)[i][X] * width + gitterx, padding + (*trainingData)[i][Y] * height + gittery, 2, bmuColor);
                }
                else
                  image->draw_circle(padding + (*trainingData)[i][X] * width + gitterx, padding + (*trainingData)[i][Y] * height + gittery, 1, contour);
            }

        //plot dataVector
        if (dataVector != NULL) {
            NodeW *bmu = som.getWinner(*dataVector);

            int nx = bmu->w[X] * image->width();
            int ny = bmu->w[Y] * image->height();
            int dx = (*dataVector)[X] * image->width();
            int dy = (*dataVector)[Y] * image->height();

            int r, g, b;
            int size = som.size()-1;
            if (size==0) size = 1;
            int h = HUE_START + bmu->getId()*MAX_HUE / (size);
            HSVtoRGB(&r, &g, &b, h, 255, 255);
            bmuColor[0] = r;
            bmuColor[1] = g;
            bmuColor[2] = b;

            image->draw_circle(padding + (*dataVector)[X] * width, padding + (*dataVector)[Y] * height, 3, bmuColor);
            image->draw_line(padding + nx, padding + ny, padding + dx, padding + dy, bmuColor);
        }

        //Draw centers
        if (drawNodes) {        
            MatMatrix<float> centers;
            som.outputCenters(centers);
            for (NodeW *bmu = som.getFirstNode(); !som.finished(); bmu = som.getNextNode()) {
                int r, g, b;
                int size = som.size()-1;
                if (size==0) size = 1;
                int h = HUE_START + bmu->getId()*MAX_HUE / (size);
                HSVtoRGB(&r, &g, &b, h, 255, 255);
                bmuColor[0] = r;
                bmuColor[1] = g;
                bmuColor[2] = b;

                image->draw_circle(padding + bmu->w[X] * width, padding + bmu->w[Y] * height, 4, contour);
                image->draw_circle(padding + bmu->w[X] * width, padding + bmu->w[Y] * height, 3, bmuColor);
                int cx = bmu->w[X] * width;
                int cy = bmu->w[Y] * height;
                int x0 = bmu->w[X] * width;
                int x1 = bmu->w[X] * width;
                int y0 = bmu->w[Y] * height;
                int y1 = bmu->w[Y] * height;
                image->draw_line(padding + cx, padding + y0, padding + cx, padding + y1, contour);
                image->draw_line(padding + x0, padding + cy, padding + x1, padding + cy, contour);
            }

            //draw labels
            if (averages != NULL) {
                for (int labelIndex = 0; labelIndex < averages->cols(); labelIndex++) {
                    std::stringstream sstr;
                    sstr << (*groupLabels)[labelIndex];
                    float x = averages->get(X, labelIndex);
                    float y = averages->get(Y, labelIndex);
                    image->draw_text(padding + x * width, padding + y * height, sstr.str().c_str(), contour);
                }
            }
        }
        
        //Draw connections
        if (drawConnections) {
            SOM<NodeW>::TPConnectionSet::iterator it;
            for (it = som.meshConnectionSet.begin(); it != som.meshConnectionSet.end(); it++) {
                float x0 = (*it)->node[0]->w[X];
                float y0 = (*it)->node[0]->w[Y];
                float x1 = (*it)->node[1]->w[X];
                float y1 = (*it)->node[1]->w[Y];
                image->draw_line(padding + x0 * width, padding + y0 * height, padding + x1 * width, padding + y1 * height, contour);
            }
        }
    }

    template <class T>
    void buildImage(SOM2D<DSNeuron, T> &som, int X, int Y, MatVector<float> *dataMatrix = NULL, bool clean = true) {

        unsigned char bmuColor[3];

        int width = (image->width() - 2 * padding);
        int height = (image->height() - 2 * padding);

        if (clean) image->fill(background[0], background[1], background[2]);

        //Draw data
        if (trueClustersColor && trueClusters.size() > 0)
            drawTrueClusters(X, Y);
        else
            for (int i = 0; i < trainingData->rows(); i++) {

                int gitterx = applyGitter(gitter);
                int gittery = applyGitter(gitter);
                if (bmucolor) {
                    MatVector<float> row;
                    trainingData->getRow(i, row);
                    DSNeuron bmu;
                    som.findBMU(row, bmu);

                    int nx = bmu.weights[X] * width;
                    int ny = bmu.weights[Y] * height;
                    int dx = (*trainingData)[i][X] * width;
                    int dy = (*trainingData)[i][Y] * height;

                    int r, g, b;
                    int size = (som.getSomRows() * som.getSomCols())-1;
                    if (size==0) size = 1;
                    int h = HUE_START +  ((bmu.r * som.getSomCols() + bmu.c)*MAX_HUE) / (size);
                    HSVtoRGB(&r, &g, &b, h, 255, 255);
                    bmuColor[0] = r;
                    bmuColor[1] = g;
                    bmuColor[2] = b;
                    image->draw_circle(padding + (*trainingData)[i][X] * width + gitterx, padding + (*trainingData)[i][Y] * height + gittery, 2, bmuColor);
                }
                else
                    image->draw_circle(padding + (*trainingData)[i][X] * width + gitterx, padding + (*trainingData)[i][Y] * height + gittery, 1, contour);
            }

        //Draw data othe data
        if (dataMatrix != NULL) {
            DSNeuron bmu;
            som.findBMU(*dataMatrix, bmu);

            int nx = bmu.weights[X] * width;
            int ny = bmu.weights[Y] * height;
            int dx = (*dataMatrix)[X] * width;
            int dy = (*dataMatrix)[Y] * height;

            int r, g, b;
            int size = (som.getSomRows() * som.getSomCols())-1;
            if (size==0) size = 1;
            int h = HUE_START + ((bmu.r * som.getSomCols() + bmu.c)*MAX_HUE) / (size);
            HSVtoRGB(&r, &g, &b, h, 255, 255);
            bmuColor[0] = r;
            bmuColor[1] = g;
            bmuColor[2] = b;
            image->draw_circle(padding + (*dataMatrix)[X] * width + applyGitter(gitter), padding + (*dataMatrix)[Y] * height + applyGitter(gitter), 3, bmuColor);
            image->draw_line(padding + nx, padding + ny, padding + dx, padding + dy, bmuColor);
        }
        /**/

        //Draw neurons
        if (drawNodes) {
            for (int x = 0; x < som.getSomRows(); x++)
                for (int y = 0; y < som.getSomCols(); y++) {
                    DSNeuron neuron(x, y);
                    som.getNeuron(neuron);
                    int r, g, b;
                    int size = (som.getSomRows() * som.getSomCols())-1;
                    if (size==0) size = 1;
                    int h = HUE_START + ((neuron.r * som.getSomCols() + neuron.c)*MAX_HUE) / (size);
                    HSVtoRGB(&r, &g, &b, h, 255, 255);
                    bmuColor[0] = r;
                    bmuColor[1] = g;
                    bmuColor[2] = b;

                    image->draw_circle(padding + neuron.weights[X] * width, padding + neuron.weights[Y] * height, 4, contour);
                    image->draw_circle(padding + neuron.weights[X] * width, padding + neuron.weights[Y] * height, 3, bmuColor);
                    if (neuron.dsWeights.size() > 0) {
                        int cx = neuron.weights[X] * width;
                        int cy = neuron.weights[Y] * height;
                        int x0 = neuron.weights[X] * width - neuron.dsWeights[X]*20;
                        int x1 = neuron.weights[X] * width + neuron.dsWeights[X]*20;
                        int y0 = neuron.weights[Y] * height - neuron.dsWeights[Y]*20;
                        int y1 = neuron.weights[Y] * height + neuron.dsWeights[Y]*20;
                        image->draw_line(padding + cx, padding + y0, padding + cx, padding + y1, contour);
                        image->draw_line(padding + x0, padding + cy, padding + x1, padding + cy, contour);
                    }
                }

            if (averages != NULL) { //draw labels
                for (int label = 0; label < averages->cols(); label++) {
                    stringstream sstr;
                    sstr << groupLabels->at(label);
                    float x = averages->get(X, label);
                    float y = averages->get(Y, label);
                    image->draw_text(padding + x*width, padding + y*height, sstr.str().c_str(), contour);
                }
            }
        }

        //Draw connections
        if (drawConnections) {
        for (int x = 0; x < som.getSomRows(); x++)
            for (int y = 0; y < som.getSomCols(); y++) {
                DSNeuron neuron(x, y);
                som.getNeuron(neuron);
                if (x > 0) {
                    DSNeuron neighbor(x - 1, y);
                    som.getNeuron(neighbor);
                    float dist = neuron.dsWeights.dist(neighbor.dsWeights) / neuron.dsWeights.size();
                    unsigned char conColor[] = {255, 255, 255};
                    conColor[0] = conColor[0] * dist;
                    conColor[1] = conColor[1] * dist;
                    conColor[2] = conColor[2] * dist;
                    image->draw_line(padding + neuron.weights[X] * width, padding + neuron.weights[Y] * height, padding + neighbor.weights[X] * width, padding + neighbor.weights[Y] * height, conColor);
                }

                if (x < som.getSomRows() - 1) {
                    DSNeuron neighbor(x + 1, y);
                    som.getNeuron(neighbor);
                    float dist = neuron.dsWeights.dist(neighbor.dsWeights) / neuron.dsWeights.size();
                    unsigned char conColor[] = {255, 255, 255};
                    conColor[0] = conColor[0] * dist;
                    conColor[1] = conColor[1] * dist;
                    conColor[2] = conColor[2] * dist;
                    image->draw_line(padding + neuron.weights[X] * width, padding + neuron.weights[Y] * height, padding + neighbor.weights[X] * width, padding + neighbor.weights[Y] * height, conColor);
                }

                if (y > 0) {
                    DSNeuron neighbor(x, y - 1);
                    som.getNeuron(neighbor);
                    float dist = neuron.dsWeights.dist(neighbor.dsWeights) / neuron.dsWeights.size();
                    unsigned char conColor[] = {255, 255, 255};
                    conColor[0] = conColor[0] * dist;
                    conColor[1] = conColor[1] * dist;
                    conColor[2] = conColor[2] * dist;
                    image->draw_line(padding + neuron.weights[X] * width, padding + neuron.weights[Y] * height, padding + neighbor.weights[X] * width, padding + neighbor.weights[Y] * height, conColor);
                }

                if (y < som.getSomCols() - 1) {
                    DSNeuron neighbor(x, y + 1);
                    som.getNeuron(neighbor);
                    float dist = neuron.dsWeights.dist(neighbor.dsWeights) / neuron.dsWeights.size();
                    unsigned char conColor[] = {255, 255, 255};
                    conColor[0] = conColor[0] * dist;
                    conColor[1] = conColor[1] * dist;
                    conColor[2] = conColor[2] * dist;
                    image->draw_line(padding + neuron.weights[X] * width, padding + neuron.weights[Y] * height, padding + neighbor.weights[X] * width, padding + neighbor.weights[Y] * height, conColor);
                }
            }
        }
    }

    void drawTrueClusters(int X, int Y) {
        unsigned char color[3];
        int r, g, b;
        int width = (image->width() - 2 * padding);
        int height = (image->height() - 2 * padding);

        int size = trueClusters.size();
        if (maxDataPlot>0)
            size = maxDataPlot;
        
        int start = 0, end = size, inc = 1;
        
        //swich class print order in 50% of the time
        if (rand()%2==0) {
            start = size-1; end = -1; inc = -1;
        }
        
        for (int k = start; k != end; k+=inc) {
            int c = k;
            if (maxDataPlot>0)
                c = rand()%size;
            
            int h = HUE_START + c * MAX_HUE / (size);
            HSVtoRGB(&r, &g, &b, h, 255, 255);
            color[0] = r;
            color[1] = g;
            color[2] = b;

            std::vector<int> &indices = trueClusters[c];
            for (int i = 0; i < indices.size(); i++) {
                int gitterx = applyGitter(gitter);
                int gittery = applyGitter(gitter);

                MatVector<float> sample;
                trainingData->getRow(indices[i], sample);
                image->draw_circle(padding + sample[X] * width + gitterx, padding + sample[Y] * height + gittery, 2, color);
            }
        }
    }

    int applyGitter(int gitter) {

        if (gitter==0)
            return 0;
        
        float r = ((float) rand() / (RAND_MAX - 1));
        return (gitter * r - gitter / 2);
    }

    void HSVtoRGB(int *r, int *g, int *b, int h, int s, int v) {
        int f;
        long p, q, t;

        h = h%360;
        
        if (s == 0) {
            *r = *g = *b = v;
            return;
        }

        f = ((h % 60)*255) / 60;
        h /= 60;

        p = (v * (256 - s)) / 256;
        q = (v * (256 - (s * f) / 256)) / 256;
        t = (v * (256 - (s * (256 - f)) / 256)) / 256;

        switch (h) {
            case 0:
                *r = v;
                *g = t;
                *b = p;
                break;
            case 1:
                *r = q;
                *g = v;
                *b = p;
                break;
            case 2:
                *r = p;
                *g = v;
                *b = t;
                break;
            case 3:
                *r = p;
                *g = q;
                *b = v;
                break;
            case 4:
                *r = t;
                *g = p;
                *b = v;
                break;
            default:
                *r = v;
                *g = p;
                *b = q;
                break;
        }
    }

};

#endif	/* DATADISPLAY_H */

