#include "TextToPhoneme.h"

#include <fstream>
#include <iostream>
#include <cfloat>

using namespace std;

const string unkwn_word = UNKWN_WORD;

TextToPhoneme::TextToPhoneme() {
}

void TextToPhoneme::loadDictionary(const string& dictionaryFile) throw (TranslationException&) {
    ifstream file(dictionaryFile.c_str());
    string word, phoneme;

    dbgOut(2) << "Dictionary:" + dictionaryFile << ":" << endl;

    if (!file.is_open())
        throw TranslationException("Could not open dictionary file:" + dictionaryFile);

    while (!file.eof()) // ler ate ao final de ficheiro
    {
        file >> word;
        file.get();
        getline(file, phoneme);

        dbgOut(3) << word << "-> '" << phoneme << "'" << endl;

        mapDictionary[word] = phoneme;
    }
    file.close();
}

TextToPhoneme::~TextToPhoneme() {
    //dtor
}

string& toupper(string &text) {
    for (unsigned int i = 0; i < text.length(); i++)
        text[i] = toupper(text[i]);

    return text;
}

bool isSeparator(char ch) {
    switch (ch) {
        case ' ':
        case '-':
        case '_':
        case '\t':
        case '\0': return true;
        default: return false;
    }
}

bool isPunctuaion(char ch) {
    switch (ch) {
        case ',':
        case '.':
        case ':':
        case '?':
        case '!': return true;

        default: return false;
    }
}

bool TextToPhoneme::getRandomWord(int wordsNum, string &phonemes) {
    string word = "", phoneme = "";
    phonemes = "";
    std::map<std::string, std::string>::iterator it;
    bool ok = true;


    for (int i = 0; i < wordsNum; i++) {
        it = mapDictionary.begin();
        std::advance(it, rand() % mapDictionary.size());
        if (it != mapDictionary.end()) //if word is found
        {
            phoneme = (*it).second; // translate word to phoneme
            dbgOut(0) << word << "-> '" << phoneme << "'" << endl;
            phonemes += phoneme + ' '; //append to phonemes
        } else { //otherwise
            phonemes += unkwn_word + ' '; //append unkonwn word
            dbgOut(1) << "Unknown word: " << word << endl;
            ok = false;
        }
    }

    return ok;
}

bool TextToPhoneme::translateWords(const string& inputText, string &phonemes) {
    string word = "", phoneme = "";
    string text = inputText + " ";
    phonemes = "";
    std::map<std::string, std::string>::iterator it;
    bool ok = true;

    for (unsigned int i = 0; i < text.size(); i++) {
        if (isSeparator(text[i]) || isPunctuaion(text[i])) {//word finished

            it = mapDictionary.find(toupper(word));
            if (it != mapDictionary.end()) //if word is found
            {
                phoneme = (*it).second; // translate word to phoneme
                dbgOut(2) << word << "-> '" << phoneme << "'" << endl;
                phonemes += phoneme + ' '; //append to phonemes
            } else { //otherwise
                phonemes += unkwn_word + ' '; //append unkonwn word
                dbgOut(1) << "Unknown word: " << word << endl;
                ok = false;
            }

            //find next word

            if (isPunctuaion(text[i])) {
                word = text[i];
            } else {
                while (isSeparator(text[i]) && i < text.size()) i++;
                i--;
                word = "";
            }
        } else
            word += text[i];
    }

    return ok;
}

PhonemesToFeatures::PhonemesToFeatures() {
}

void PhonemesToFeatures::Clear() {
    std::map<std::string, Features *>::iterator it;

    for (it = mapFeatures.begin(); it != mapFeatures.end(); it++) {
        delete (*it).second;
    }

    mapFeatures.erase(mapFeatures.begin(), mapFeatures.end());
}

PhonemesToFeatures::~PhonemesToFeatures() {
    Clear();
}

void PhonemesToFeatures::loadPhonemeFeatures(const std::string& featuresFile, int numFeatures) throw (TranslationException&) {
    ifstream file(featuresFile.c_str());
    string phoneme;
    float feature;
    Features *features;

    dbgOut(2) << "Features:" + featuresFile << ":" << endl;

    if (!file.is_open())
        throw TranslationException("Could not open features file:" + featuresFile);

    //skip first row
    getline(file, phoneme);

    //delete old features
    Clear();

    //read features
    while (!file.eof()) // ler ate ao final de ficheiro
    {
        file >> phoneme;
        features = new Features(numFeatures);
        for (int i = 0; i < numFeatures; i++) {
            file >> feature;
            (*features)[i] = feature;
        }
        dbgOut(2) << phoneme << ": " << features->size() << endl;
        mapFeatures[toupper(phoneme)] = features;
    }
    file.close();

}

void PhonemesToFeatures::translatePhonemesFeatures(const std::string &phonemes, FeaturesVector &featuresVector) {
    Features *features;
    string phoneme = "";

    phonemesVector.clear();
    featuresVector.size(0, 0);

    for (unsigned int i = 0; i < phonemes.size(); i++) {
        if (isSeparator(phonemes[i]) || isPunctuaion(phonemes[i])) {//word finished

            features = mapFeatures[phoneme]; // translate phoneme to features
            if (features != NULL) {
                if (featuresVector.rows() != features->size())
                    featuresVector.size(features->size(), 0);

                featuresVector.concatCols(*features);
                dbgOut(3) << phoneme << ": " << features->size() << endl;
                phonemesVector.push_back(phoneme);
            } else
                dbgOut(1) << phoneme << ": feature NULL" << endl;

            //go to next word
            while ((isPunctuaion(phonemes[i]) || isSeparator(phonemes[i])) && i < phonemes.size()) i++;
            i--;
            phoneme = "";
        } else
            phoneme += phonemes[i];
    }
}

void PhonemesToFeatures::getPhoneme(unsigned int i, std::string &phoneme) {
    if (i < phonemesVector.size())
        phoneme = phonemesVector[i];
    else
        phoneme = "";
}

void PhonemesToFeatures::getRandomPhoneme(std::string &phoneme) {
    vector<std::string> vector;
    for (std::map<std::string, Features*>::iterator it = mapFeatures.begin(); it != mapFeatures.end(); ++it) {
        vector.push_back(it->first);
    }

    phoneme = vector[random() % vector.size()];
    
}

float PhonemesToFeatures::translateFeaturesPhoneme(Features &features, std::string &phoneme) {

    float dist, minDist = FLT_MAX;
    std::map<std::string, Features*>::iterator it;
    Features *featuresMap;

    for (it = mapFeatures.begin(); it != mapFeatures.end(); it++) {
        featuresMap = it->second;
        if (featuresMap != NULL) {
            dist = 0;
            for (uint i = 0; i < featuresMap->size(); i++) {
                dist += qrt((*featuresMap)[i] - features[i]);
            }
            if (dist < minDist) {
                minDist = dist;
                phoneme = it->first;
            }
        }
    }

    return sqrt(minDist);
}

