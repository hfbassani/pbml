#ifndef TEXTTOPHONEME_H
#define TEXTTOPHONEME_H

#include "Defines.h"
#include "MatMatrix.h"
#include "MatVector.h"
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <exception>

#define UNKWN_WORD "UKWN"

class TranslationException : public std::exception {
public:
    std::string strmsg;

    TranslationException(const std::string &msg) {
        this->strmsg = msg;
    }

    ~TranslationException() throw () {
    }

    virtual const char* what() const throw () {
        return strmsg.c_str();
    }
};

class TextToPhoneme {
    std::map<std::string, std::string> mapDictionary;

public:

    TextToPhoneme();
    virtual ~TextToPhoneme();

    void loadDictionary(const std::string &dictionaryFile) throw (TranslationException&);
    bool translateWords(const std::string &text, std::string &phonemes);
    bool getRandomWord(const int wordsNum, std::string &phonemes);
};

typedef MatVector<float> Features;
typedef MatMatrix<float> FeaturesVector;

class PhonemesToFeatures {
    std::map<std::string, Features*> mapFeatures;
    std::vector<std::string> phonemesVector;

public:

    PhonemesToFeatures();
    ~PhonemesToFeatures();
    void Clear();

    void loadPhonemeFeatures(const std::string &featuresFile, int numFeatures) throw (TranslationException&);
    void translatePhonemesFeatures(const std::string &phonemes, FeaturesVector &featuresVector);
    float translateFeaturesPhoneme(Features &features, std::string &phoneme);
    void getPhoneme(unsigned int i, std::string &phoneme);
    void getRandomPhoneme(std::string &phoneme);
};

#endif // TEXTTOPHONEME_H
