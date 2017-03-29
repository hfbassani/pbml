/* 
 * File:   LHSParameters.h
 * Author: hans
 *
 * Created on 14 de Julho de 2014, 18:40
 */

#ifndef LHSPARAMETERS_H
#define	LHSPARAMETERS_H

#include "Parameters.h"
#include <vector>
#include <cstdlib>
#include <string>

bool getRangeFromLine(std::string const line, std::string &range);

class LHSParameter: public Parameter<double> 
{
    double minValue;
    double maxValue;
    std::vector<double> perm;
    bool rangeSet;

public:
    static const std::string rangeStartStr; //separador inicial de range
    static const std::string rangeEndStr; //separador inicial de range
    static const std::string rangeSep; //separador de range
    
    LHSParameter() {
        rangeSet = false;
    }

    inline LHSParameter& operator=(const double& value)
    {
            this->value = value;

            if (rangeSet && (value<minValue || value>maxValue)) {
                std::cerr << "Parameter '" << this->name << "' is out of range: " << value << " is not in [" << minValue << "," << maxValue << "]." << std::endl;
            }

            return *this;
    }
        
    inline LHSParameter& setRange(const double &vmin, const double &vmax) {
        minValue = vmin;
        maxValue = vmax;
        rangeSet = true;
        return *this;
    }
    
    inline LHSParameter& clearRange() {
        rangeSet = false;
        return *this;
    }
        
    virtual std::string toString() const {

            std::ostringstream ost;
            ost << this->name << "=" << this->value << CFGFile::endValueStr;

            if (rangeSet) {
                ost << "\t" << rangeStartStr << minValue << rangeSep << maxValue << rangeEndStr;
            }

            ost << "\t" << CFGFile::commentStr << this->description;
            return ost.str();
    }
    
    virtual void fromString(std::string& str) {

            if (!getIdFromLine(str, this->name))
                    std::cerr << "Error reading parameter name " << std::endl;

            bool gotValue = false;

            std::string valueStr;
            if (getValueFromLine(str, valueStr))
            {
                    std::istringstream ist(valueStr);
                    ist >> this->value;
                    gotValue = true;
            }

            std::string vrange;
            if (getRangeFromLine(str, vrange)) {
                std::istringstream ist(vrange);
                char coma;
                ist >> minValue;

                while (!ist.eof()) {
                    ist >> coma;
                    ist >> maxValue;
                    if (!ist.bad() || !ist.fail()) break;
                }

                if (!gotValue) this->value = minValue;
                gotValue = true;
                rangeSet = true;
            }

            if (!gotValue) 
                std::cerr << "Error reading value for parameter " << this->name << std::endl;
            else
            if (rangeSet && (this->value<minValue || this->value > maxValue)) //value is out of range
                std::cerr << "Value is out or range for parameter " << this->name << std::endl;

            getDescriptionFromLine(str, this->description);
    }    
        
    void makeRandomValueList(int N) {
        
        double step = (maxValue - minValue)/(double)N;

        this->perm.resize(N);
        int i, k;
        // initialization
        for (i = 0; i < N; i++)
                this->perm[i] = (double)(minValue + i*step + nextRand()*step);
        
        // permutation
        for (i = 0; i < N-1; i++)
        {
                 k = i + (int)(nextRand()*(N-i));
                 double temp = perm[i];
                 perm[i] = perm[k];
                 perm[k] = temp;
        }
    }
    
    void selectRandomValue(int i) {
        this->value = perm[i];
    }
    
    double nextRand() {
        return (double)rand() / RAND_MAX;
    }
};

class LHSParameters: public Parameters {
private:
    typedef std::list<LHSParameter *> LHSParameterList;
    LHSParameterList lhsParams;
    int i;
    int N;
    
public:
    #define addParameterToLHS(parameter) addParameterToLHSn(parameter, #parameter)
    #define addParameterToLHSd(parameter, description) addParameterToLHSnd(parameter, #parameter, description)
    #define addParameterToLHSdv(parameter, description, value) addParameterToLHSndv(parameter, #parameter, description, value)
    #define addParameterToLHSdr(parameter, description, min, max) addParameterToLHSndr(parameter, #parameter, description, min, max)
    #define addParameterToLHSdvr(parameter, description, value, min, max) addParameterToLHSndvr(parameter, #parameter, description, value, min, max)
    void addParameterToLHSn(LHSParameter &parameter, std::string name);
    void addParameterToLHSnd(LHSParameter &parameter, std::string name, std::string description);
    void addParameterToLHSndv(LHSParameter &parameter, std::string name, std::string description, double value);
    void addParameterToLHSndr(LHSParameter &parameter, std::string name, std::string description, double min, double max);
    void addParameterToLHSndvr(LHSParameter &parameter, std::string name, std::string description, double value, double min, double max);
    void clearLHS();
    void initLHS(int N);
    void initLHS(int N, unsigned long seed);
    bool setNextValues();
    bool finished();
    int size() {return N;};
};

#endif	/* LHSPARAMETERS_H */

