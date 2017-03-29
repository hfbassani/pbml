/* 
 * File:   LHSParameters.cpp
 * Author: hans
 * 
 * Created on 14 de Julho de 2014, 18:40
 */

#include "LHSParameters.h"
#include <string>

using namespace std;

const string LHSParameter::rangeStartStr = "[";
const string LHSParameter::rangeEndStr = "]";
const string LHSParameter::rangeSep = ",";

void LHSParameters::addParameterToLHSn(LHSParameter &parameter, std::string name) {
    lhsParams.push_back(&parameter); 
    addParameterN(parameter, name);
}

void LHSParameters::addParameterToLHSnd(LHSParameter &parameter, std::string name, std::string description) {
    lhsParams.push_back(&parameter);
    addParameterND(parameter, name, description);
}

void LHSParameters::addParameterToLHSndv(LHSParameter &parameter, std::string name, std::string description, double value) {
    lhsParams.push_back(&parameter);
    addParameterNDV(parameter, name, description, value);
}

void LHSParameters::addParameterToLHSndr(LHSParameter &parameter, std::string name, std::string description, double min, double max) {
    lhsParams.push_back(&parameter);
    parameter.setRange(min, max);
    addParameterND(parameter, name, description);
}

void LHSParameters::addParameterToLHSndvr(LHSParameter &parameter, std::string name, std::string description, double value, double min, double max) {
    lhsParams.push_back(&parameter);
    parameter.setRange(min, max);
    addParameterNDV(parameter, name, description, value);
}

void LHSParameters::clearLHS() {
    lhsParams.clear();
}

void LHSParameters::initLHS(int N, unsigned long seed) {
    srand(seed);
    initLHS(N);
}

void LHSParameters::initLHS(int N) {
    i = 0;
    this->N = N;
    
    LHSParameterList::iterator it;
    for (it = lhsParams.begin(); it!=lhsParams.end(); it++) {
        (*it)->makeRandomValueList(N);
        (*it)->selectRandomValue(i);
    }
}

bool LHSParameters::setNextValues() {
    
    if (finished()) return false;    
    i++;
    
    LHSParameterList::iterator it;
    for (it = lhsParams.begin(); it!=lhsParams.end(); it++) {
        (*it)->selectRandomValue(i);
    }
    
    return !finished();
}

bool LHSParameters::finished() {
    return (i>=N);
}

bool getRangeFromLine(string const line, string &range)
{
    size_t delimiterEnd = line.find(LHSParameter::rangeStartStr) + LHSParameter::rangeStartStr.length();
    size_t endValueStart = line.find(LHSParameter::rangeEndStr, delimiterEnd);
    if (delimiterEnd < string::npos && endValueStart < string::npos)
    {
            range = line.substr(delimiterEnd, endValueStart-delimiterEnd);
            trim(range);
            if (range.length()>0)
                    return true;
    }

    return false;
}