/* 
 * File:   RunCrossSTest.h
 * Author: hans
 *
 * Created on 18 de Novembro de 2013, 08:19
 */

#ifndef CROSSSTEST_H
#define	CROSSSTEST_H

#include <map>
#include "TextToPhoneme.h"
#include "GDSSOMMW.h"
#include "ART2Context.h"

#define STR_SG_MAP "#<map>"
#define STR_SG_TRAIN "#<train>"
#define STR_SG_TRAIN_CONTEXT "#<train-context>"
#define STR_SG_TEST "#<test>"
#define STR_SG_SEQUENCE "#<sequence-test>"
#define STR_SG_SEQUENCE_CONTEXT "#<sequence-test-context>"

#define STR_SG_RESET "#<reset>"
#define STR_SG_NAME "#<exp-name>"
#define STR_WORD_SPLIT "#<word-split>"
#define STR_SG_RESULTS "#<print-results>"

#define STR_TI_SINGLE "#<single>"
#define STR_TI_DOUBLE "#<double>"
#define STR_TI_NOISE "#<noise>"

#define STR_TT_FIRST "#<first>"
#define STR_TT_RANK "#<rank>"

#define STR_LOG_PRINT "#<print>"
#define STR_LOG_INTEREST "#<interest-word>"
#define STR_LOG_VAR "#<log-variables>"

#define DEFAULT_DICTIONARY "cmu-dict.csv"
#define DEFAULT_FEATURES "PhonemeFeatures.csv"

class TestMapping {
public:
    std::string testWord;
    std::string imageWord;
    std::string imageName;
    float activation;
    float v_act;
    float a_act;
    float c_act;
    float v_ds;
    float a_ds;
    float c_ds;
    int nodeIndex;
};

enum TlastTest {TLT_CORRECT, TLT_INCORRECT, TLT_NONE};
class TImageFeatures {
public:
    std::string word;
    MatVector<float> audioActivation;
    MatVector<float> visualActivation;    
    TlastTest lastCorrect;
};

class CrossSituationalTest {
    enum TStage {SG_UNDEF, SG_MAP, SG_TRAIN, SG_TEST, SG_SEQ, SG_NAME};
    enum TTestInstance {TI_SINGLE, TI_DOUBLE, TI_NOISE};
    enum TTestType {TT_FIRST, TT_RANK};
    
    TStage stage;
    TTestInstance testInstance;
    TTestType testType;
    std::string expName;
    std::map<std::string, TImageFeatures> wordMap;
    std::map<int, std::vector<std::string> > nodeMap;
    std::map<std::string, vector<float> > mapResults;
    std::string logFileName;
    std::string filePath;
    std::ofstream logFile;
    std::string interestWord;
    int wordSplit;
    std::vector<std::string> logVariables;
    
    TextToPhoneme ttp;
    PhonemesToFeatures pf;
    GDSSOMMW *audioCB;
    GDSSOMMW *visualCB;
    GDSSOMMW *objectMap;
    ART2Context *context;
    std::string imagePath;
    
    bool isContext;
    float singleCorrect;
    int nSingle;
    
    float bothCorrect;
    float eitherCorrect;
    float firstCorrect;
    int nDouble;    
    
    int incorrectCorrect;
    int correctCorrect;
    int firstGuess;
    int totalTest;
    
    int correctWordContext;
    int totalContext;
    
    int v_size;
    int a_size;
    int c_size;
        
    bool normalizeInputs;
public:
    CrossSituationalTest(GDSSOMMW *audioCB, GDSSOMMW *visualCB, GDSSOMMW *objectMap, ART2Context *context, const std::string &imagePath);
    void setNormalizeInputs();    
    bool runCSTest(const std::string &filename);
    virtual ~CrossSituationalTest();
    void printMap(const std::vector<TestMapping> &mappings, const std::string &trainingWord);
    void printResults();
    void reset();
    void setLogFile(std::string logFileName);
    void printLogHeader();
    void setLogVariables(std::string line);
    void logResults();
    void closeLogFile();
    void getAverageSTD(const std::string &resultName, double &average, double &std);
    void clearResults();
    std::string &getExpName();
private:
    bool addLineToMap(const std::string &line);
    bool train(const std::string &line);
    bool test(const std::string &line);
    bool seqTest(const std::string &line);    
    bool trainContext(const std::string &line);    
    
    bool translateString(std::string str, MatMatrix<float> &audioInput);
    void getAudioActivation(MatMatrix<float> &audioInput, MatVector<float> &activationAudio);
    
    bool getImageInput(std::string word, MatMatrix<float> &visualInput);
    void getImageActivation(MatMatrix<float> &visualInput, MatVector<float> &activationVisual);
    
    MatVector<float> getContext();
    
    void printDebug(GDSSOMMW::TNode *winner, const MatVector<float> &input, std::string &word, std::string &img);
    void computeActivations(GDSSOMMW::TNode *winner, TestMapping &mapping, const MatVector<float> &input);
    
    bool isInterestWord(const std::string &word);
};

#endif	/* CROSSSTEST_H */

