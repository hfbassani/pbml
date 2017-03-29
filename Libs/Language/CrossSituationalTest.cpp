/* 
 * File:   RunCrossSTest.cpp
 * Author: hans
 * 
 * Created on 18 de Novembro de 2013, 08:19
 */

#include <string>
#include <vector>

#include <fstream>
#include <numeric>
#include <algorithm>
#include "CrossSituationalTest.h"
#include "DebugOut.h"
#include "StringHelper.h"
#include "ArffData.h"
#include "GDSSOM.h"
#include "ART2Context.h"

CrossSituationalTest::CrossSituationalTest(GDSSOMMW *audioCB, GDSSOMMW *visualCB, GDSSOMMW *objectMap, ART2Context *context, const std::string &imagePath) {

    this->audioCB = audioCB;
    this->visualCB = visualCB;
    this->objectMap = objectMap;
    this->context = context;
    this->imagePath = imagePath + "/";   
    interestWord = "*";
    a_size = v_size = c_size = 0;
    wordSplit = -1;
    normalizeInputs = false;
    stage = SG_UNDEF;

    ttp.loadDictionary(DEFAULT_DICTIONARY);
    pf.loadPhonemeFeatures(DEFAULT_FEATURES, 12);
    reset();
}

void CrossSituationalTest::setNormalizeInputs() {
    normalizeInputs = true;
    context->parameters.denorm_weight = 0;
}

void CrossSituationalTest::reset() {

    isContext = false;
    singleCorrect = 0;
    nSingle = 0;

    bothCorrect = 0;
    firstCorrect = 0;
    eitherCorrect = 0;
    nDouble = 0;

    correctCorrect = 0;
    incorrectCorrect = 0;
    firstGuess = 0;
    totalTest = 0;
    
    correctWordContext = 0;
    totalContext = 0;

    std::map<std::string, TImageFeatures>::iterator it;

    for (it = wordMap.begin(); it!=wordMap.end(); it++) {
        (*it).second.lastCorrect = TLT_NONE;
    }

    objectMap->reset();
    context->init(1);

    stage = SG_UNDEF;
}

void CrossSituationalTest::printResults() {
    
    if (stage==SG_TEST) {

        float resSingle = singleCorrect/nSingle;

        dbgOut(1) << endl << expName << ":" << endl;
        dbgOut(1) << "Single: " << singleCorrect << "/" << nSingle << ":\t" << resSingle << endl;
        mapResults[expName + "-" + "single"].push_back(resSingle);

        if (nDouble>0) {
             float resEither = eitherCorrect/nDouble;
             mapResults[expName + "-" + "either"].push_back(resEither);
             float resBoth = bothCorrect/nDouble;
             mapResults[expName + "-" + "both"].push_back(resBoth); 
             float resEarlyFirst = firstCorrect/(float)nDouble;
             mapResults[expName + "-" + "early"].push_back(resEarlyFirst); 
             float resLatefirst = (bothCorrect-firstCorrect)/(float)nDouble;
             mapResults[expName + "-" + "late"].push_back(resLatefirst); 

             dbgOut(1) << "Either: " << eitherCorrect << "/" << nDouble << ":\t" << resEither << endl;

             dbgOut(1) << "Both: " << bothCorrect << "/" << nDouble << ":\t" << resBoth << endl;

             dbgOut(1) << "Early first: " << firstCorrect << "/" << bothCorrect << ":\t" << resEarlyFirst << endl;

             dbgOut(1) << "Late first: " << (bothCorrect-firstCorrect) << "/" << bothCorrect << ":\t" << resLatefirst << endl;
        }
        
        if (totalContext>0) {
            float contextRate = correctWordContext/(float)totalContext;
            dbgOut(1) << "Correct context: " << correctWordContext << "/" << totalContext << ":\t" << contextRate << endl;
            mapResults[expName + "-" + "context"].push_back(contextRate);
        }
        
        incorrectCorrect = 0;
        correctCorrect = 0;
        
   } else
   if (stage == SG_SEQ) {

        float correct = firstGuess + incorrectCorrect + correctCorrect;
        
        dbgOut(1) << endl << "Total correct:\t" << correct << "\t" << totalTest << "\t" << correct/(float)totalTest << "" << endl;   
        
        if (firstGuess>0)
                dbgOut(1) << endl << "First guess:\t" << firstGuess << endl;
        
        if (correct>0) {
            dbgOut(1) << "Last incorrect:\t" << incorrectCorrect << "\t" << correct << "\t" << incorrectCorrect/correct << "" << endl;
            dbgOut(1) << "Last correct:\t" << correctCorrect << "\t" << correct << "\t" << correctCorrect/correct << "" << endl;   
        } else {
            dbgOut(1) << "Last incorrect:\t0\t0\t0" << endl;
            dbgOut(1) << "Last correct:\t0\t0\t0" << endl;
        }
        
        logFile << correctCorrect << "\t" << incorrectCorrect << "\t" << correct << "\t"; 

        incorrectCorrect = 0;
        correctCorrect = 0;
        firstGuess = 0;
        totalTest = 0;        
   }
    
   dbgOut(1) << "Total nodes: " << objectMap->size() << endl;
   if (isContext) {
        dbgOut(1) << "Total contexts: " << context->Y.size() << endl;
   }
   
   //dbgOut(0) << objectMap->size() << "\t";
}


void CrossSituationalTest::setLogVariables(std::string line) {
    logVariables = split(line, '\t');
}

void CrossSituationalTest::setLogFile(std::string logFileName) {
    this->logFileName = logFileName;
    logVariables = split("Yu2007-Exp1-2x2-single\tYu2007-Exp1-3x3-single\tYu2007-Exp1-4x4-single\tYurokovisky2013-Exp1-single\tYurokovisky2013-Exp1-either\tYurokovisky2013-Exp1-both\tYurokovisky2013-Exp2-single\tYurokovisky2013-Exp2-either\tYurokovisky2013-Exp2-both\tYurokovisky2013-Exp3-single\tYurokovisky2013-Exp3-either\tYurokovisky2013-Exp3-both\tYurokovisky2013-Exp3-early\tYurokovisky2013-Exp3-late\tContext-1Word-context\tContext-2Word-context\tContext-4Word-context\tContext-6Word-context", '\t');
}

void CrossSituationalTest::printLogHeader() {
    logFile << "Exp\ta_t\tlp\te_b\te_n\tdsbeta\te_ds\tmwd\taw";
    for (int i=0; i<logVariables.size(); i++) {
        logFile << "\t" << logVariables[i];
    }
    logFile << endl;
}

void CrossSituationalTest::logResults() {
    
   if (!logFile.is_open()) {
        logFile.open(logFileName.c_str());
        printLogHeader();
   }
    
   double average, std;
   logFile << "All" << "\t" << objectMap->a_t << "\t" << objectMap->lp << "\t" << objectMap->e_b << "\t" << objectMap->e_n << "\t" << objectMap->dsbeta << "\t" << objectMap->epsilon_ds << "\t" << objectMap->minwd << "\t" << objectMap->age_wins;
   
   if (logVariables.size()<1) {
        getAverageSTD("Yu2007-Exp1-2x2-single", average, std);
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Yu2007-Exp1-3x3-single", average, std);
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Yu2007-Exp1-4x4-single", average, std);
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Yurokovisky2013-Exp1-single", average, std);
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Yurokovisky2013-Exp1-either", average, std);
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Yurokovisky2013-Exp1-both", average, std);
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Yurokovisky2013-Exp2-single", average, std);
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Yurokovisky2013-Exp2-either", average, std);
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Yurokovisky2013-Exp2-both", average, std);   
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Yurokovisky2013-Exp3-single", average, std);
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Yurokovisky2013-Exp3-either", average, std);
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Yurokovisky2013-Exp3-both", average, std);
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Yurokovisky2013-Exp3-early", average, std);
     //   logFile << "\t" << std;
       logFile << "\t" << average;
        getAverageSTD("Yurokovisky2013-Exp3-late", average, std);
     //   logFile << "\t" << std;
        logFile << "\t" << average;
        getAverageSTD("Context-1Word-context", average, std);
     //   logFile << "\t" << std;
       logFile << "\t" << average;
        getAverageSTD("Context-2Word-context", average, std);
     //   logFile << "\t" << std;
       logFile << "\t" << average;
       getAverageSTD("Context-4Word-context", average, std);
     //   logFile << "\t" << std;
       logFile << "\t" << average;
        getAverageSTD("Context-6Word-context", average, std);
     //   logFile << "\t" << std;
       logFile << "\t" << average;
   } else {
       for (int i=0; i<logVariables.size(); i++) {
           getAverageSTD(logVariables[i], average, std);
//           logFile << "\t" << std;
           logFile << "\t" << average;
       }
   }
  
   logFile << endl;   
}

void CrossSituationalTest::closeLogFile() {
    logFile.close();
}

void CrossSituationalTest::getAverageSTD(const std::string &resultName, double &average, double &std) {
    
    vector<float> &v = mapResults[resultName];
    if (v.size()<1) {
        average = -1;
        std = -1;
        return;
    }
    
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    average = sum / v.size();

    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), std::bind2nd(std::minus<double>(), average));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    std = std::sqrt(sq_sum / v.size());
}

void CrossSituationalTest::clearResults(){
    mapResults.clear();
}

std::string &CrossSituationalTest::getExpName() {
    return expName;
}

bool CrossSituationalTest::runCSTest(const std::string &filename) {

   std::ifstream file(filename.c_str());
    
   if (!file.is_open()) {
         return false;
   }
   
   filePath = dirnameOf(filename) + "/";
   
   stage = SG_UNDEF;
   testInstance = TI_SINGLE;
   testType = TT_FIRST;
   
   int lineNumber = 1;
   do {
       std::string line;
       getline(file, line, '\n'); lineNumber++; trim(line);       
       if (file.eof()) break;
       if (line.length()<1) continue;
       
       if (line[0] == '#') { //change mode

           if (line.compare(STR_SG_MAP)==0) {
               dbgOut(3) << STR_SG_MAP << " flag found." << endl;
               stage = SG_MAP;
           } else if (line.compare(STR_SG_TRAIN)==0) {
               dbgOut(3) << STR_SG_TRAIN << " flag found." << endl;
               stage = SG_TRAIN;
           } else if (line.compare(STR_SG_TRAIN_CONTEXT)==0) {
               dbgOut(3) << STR_SG_TRAIN_CONTEXT << " flag found." << endl;
               isContext = true;
               stage = SG_TRAIN;
           } else if (line.compare(STR_SG_TEST)==0) {
               dbgOut(3) << STR_SG_TEST << " flag found." << endl;
               stage = SG_TEST;              
           } else if (line.compare(STR_SG_SEQUENCE)==0) {
               dbgOut(3) << STR_SG_SEQUENCE << " flag found." << endl;
               logFile << endl;
               stage = SG_SEQ;
           } else if (line.compare(STR_SG_SEQUENCE_CONTEXT)==0) {
               dbgOut(3) << STR_SG_SEQUENCE_CONTEXT << " flag found." << endl;
               logFile << endl;
               isContext = true;
               stage = SG_SEQ;
           } else if (line.compare(STR_SG_RESULTS)==0) {
               dbgOut(3) << STR_SG_RESULTS << " flag found." << endl;
               printResults();
           } else if (line.compare(STR_SG_RESET)==0) {
               dbgOut(3) << STR_SG_RESET << " flag found." << endl;
               reset();
           } else if (line.compare(STR_SG_NAME)==0) {
               dbgOut(3) << STR_SG_NAME << " flag found." << endl;
               stage = SG_NAME;
           }  else if (line.compare(STR_SG_NAME)==0) {
               dbgOut(3) << STR_SG_NAME << " flag found." << endl;
               stage = SG_NAME;
           } else if (line.compare(STR_WORD_SPLIT)==0) {
               dbgOut(3) << STR_WORD_SPLIT << " flag found." << endl;
               getline(file, line, '\n'); lineNumber++;  trim(line);
               wordSplit = atoi(line.c_str());
           } else if (line.compare(STR_TI_SINGLE)==0) {
               dbgOut(3) << STR_TI_SINGLE << " flag found." << endl;
               testInstance = TI_SINGLE;
           } else if (line.compare(STR_TI_DOUBLE)==0) {
               dbgOut(3) << STR_TI_DOUBLE << " flag found." << endl;
               testInstance = TI_DOUBLE;
           } else if (line.compare(STR_TI_NOISE)==0) {
               dbgOut(3) << STR_TI_NOISE << " flag found." << endl;
               testInstance = TI_NOISE;
           } else if (line.compare(STR_TT_FIRST)==0) {
               dbgOut(3) << STR_TT_FIRST << " flag found." << endl;
               testType = TT_FIRST;
           } else if (line.compare(STR_TT_RANK)==0) {
               dbgOut(3) << STR_TT_RANK << " flag found." << endl;
               testType = TT_RANK;
           } else if (line.compare(STR_LOG_PRINT)==0) {
               dbgOut(3) << STR_LOG_PRINT << " flag found." << endl;
               getline(file, line, '\n'); lineNumber++;
               dbgOut(0) << line << endl;
           } else if (line.compare(STR_LOG_INTEREST)==0) {
               dbgOut(3) << STR_LOG_INTEREST << " flag found." << endl;
               getline(file, line, '\n'); lineNumber++;  trim(line);
               interestWord = line;
           } else if (line.compare(STR_LOG_VAR)==0) {
               dbgOut(3) << STR_LOG_VAR << " flag found." << endl;
               getline(file, line, '\n'); lineNumber++;  trim(line);
               setLogVariables(line);
           } else {
               dbgOut(0) << "Unknown directive: " << line << endl;
           }
           
           continue;
       }
       
       switch (stage) {
           case SG_MAP:
               if (!addLineToMap(line))
                   dbgOut(1) << "fail to split word map in line " <<  lineNumber << ": " << line << endl; 
               break;
           case SG_TRAIN:
               if (!train(line)) {
                   dbgOut(1) << "fail parse train line " <<  lineNumber << ": " << line << endl; 
               }
               break;         
           case SG_TEST:
               if (!test(line)) {
                   dbgOut(1) << "fail parse test line " <<  lineNumber << ": " << line << endl; 
               }
               break;   
           case SG_SEQ:
               if (!seqTest(line)) {
                   dbgOut(1) << "fail parse sequence test line " <<  lineNumber << ": " << line << endl; 
               }
               break;                 
           case SG_NAME:
               expName = line;
               //dbgToLogFile(filePath + expName+".log");
               //dbgOut(1) << "Experiment: " << expName << endl;
               break;
       }

   } while (!file.eof());

   return true;
}

std::string removeWordType(const std::string &word) {
    int i = word.find('-');
    return word.substr(0,i);
}

bool CrossSituationalTest::addLineToMap(const std::string &line) {
    std::vector<std::string> words = split(line, '\t');
    
    if (words.size()!=2) {
        return false;
    }
    
    if (words[0].length()<1 || words[1].length() <1) {
        return false;
    }
    
    std::map<std::string, TImageFeatures>::const_iterator it = wordMap.find(words[0]);
    
    if (it!=wordMap.end()) {
        if ((*it).second.word.compare(words[1])==0)
            return true; //map already loaded
    }
    
    dbgOut(2) << "Loading map features: " << words[0] << "->" << words[1] << endl;
            
    TImageFeatures imageFeatures;
    imageFeatures.word = words[1];
    imageFeatures.lastCorrect = TLT_NONE;
    
    //audio features
    MatMatrix<float> audioInput;
    translateString(removeWordType(imageFeatures.word), audioInput);
    getAudioActivation(audioInput, imageFeatures.audioActivation);
    
    
    if (interestWord.compare(removeWordType(imageFeatures.word))==0) {
            
        dbgOut(3) << imageFeatures.word << ":" << audioInput.toString() << endl;
    }
    
    //visual features
    MatMatrix<float> visualInput;
    if (!getImageInput(imageFeatures.word, visualInput)) {
        return false;
    };
    getImageActivation(visualInput, imageFeatures.visualActivation);
    
    wordMap[words[0]] = imageFeatures;
            
    dbgOut(3) << words[0] << " maps to: " << words[1] << endl;
    return true;
}

bool CrossSituationalTest::train(const std::string &line) {

    std::vector<std::string> words = split(line, '\t');
    std::string trainText = "";
    int split = wordSplit;
    
    if (split<1)
        split = words.size()/2; //assume half-half
    
    if (words.size()<split) {
        return false;
    }
    
    for (int i=0; i<split; i++) {
        trainText = trainText + removeWordType(wordMap[words[i]].word) + " ";
    }
    dbgOut(2) << "Train words: " << trainText << endl;
    
    MatMatrix<float> audioInput;
    translateString(trainText, audioInput);    
    
    MatVector<float> audioActivation;
    getAudioActivation(audioInput, audioActivation);
    
    //randomize image prezentation
    for (int i=split; i<words.size();i++) {
        int k = split + rand()%(words.size()-split);
        
        if (i!=k) {
            std::string tmp = words[k];
            words[k] = words[i];
            words[i] = tmp;
        }
    }
    
    dbgOut(2) << "Train images:";
    for (int i=split; i<words.size(); i++) {
        dbgOut(2) << " " << words[i];
        //Visual info

        MatVector<float> input = wordMap[words[i]].visualActivation;

        v_size = input.size();
        a_size = audioActivation.size();

        //Auditory info
        input.concat(audioActivation);        

        //Context info        
        if (isContext) {
            if (context->U.size()!=input.size()) 
                context->init(input.size());

            dbgOut(0) << trainText << "\t";

            context->train(input);
            input.concat(getContext());
            c_size = getContext().size();

            dbgOut(1)  << endl;
        }

        dbgOut(3) << input.toString() << endl;

        if (objectMap->getInputSize()!=input.size())
            objectMap->reset(input.size());

        objectMap->trainningStep(input);
    }
    dbgOut(2) << endl;
    
    dbgOut(2) << "Map size: " << objectMap->size() << endl;
    
    return true;
}

MatVector<float> CrossSituationalTest::getContext() {
    MatVector<float> currentContext = context->getCurrentContext(!normalizeInputs);    
    return currentContext;
}

bool CrossSituationalTest::seqTest(const std::string &line) {

    std::vector<std::string> words = split(line, '\t');
    
    if (words.size()<1) {
        return false;
    }
    
    dbgOut(2) << "Train word: " << words[0] << endl;
    
    MatVector<float> audioActivation = wordMap[words[0]].audioActivation;
    
    dbgOut(2) << "Train images:";
    int nTrains = 1;
    std::vector<TestMapping> mappings; 
    for (int k=0;k<nTrains;k++) {

        //randomize image presentation
        for (int i=1; i<words.size();i++) {
            int k = 1 + rand()%(words.size()-1);

            if (i!=k) {
                std::string tmp = words[k];
                words[k] = words[i];
                words[i] = tmp;
            }
        }
        
        for (int i=1; i<words.size(); i++) {
            
            //Visual info
            MatVector<float> input = wordMap[words[i]].visualActivation;
            
            v_size = input.size();
            a_size = audioActivation.size();
            
            //Auditory info
            input.concat(audioActivation);        

            //Context info
            if (isContext) {
                if (context->U.size()!=input.size()) 
                    context->init(input);

                dbgOut(0) << words[i] << "\t";
                
                context->train(input);
                input.concat(getContext());
                c_size = getContext().size();
            }

            dbgOut(3) << input.toString() << endl;

            if (objectMap->getInputSize()!=input.size())
                objectMap->reset(input.size());

            objectMap->trainningStep(input);
 
            GDSSOMMW::TNode *winner = objectMap->getWinner(input);

            TestMapping mapping;
            mapping.testWord = words[0];
            mapping.imageName = wordMap[words[i]].word;
            mapping.imageWord = words[i];
            mapping.nodeIndex = winner->getId();
            computeActivations(winner, mapping, input);
//            if (winner->wins<nTrains) //New nodes do not contribuite
//                mapping.activation = 0;

            //printDebug(winner, visualActivation, words[0], words[i]);

            mappings.push_back(mapping);            
        }
    }
    dbgOut(2) << endl;
    
    printMap(mappings, words[0]);
    
    //Sord winner images
    bool ordered;
    do {
        ordered = true;
        for (int index=0; index<mappings.size()-1;index++) {
            if (mappings[index].activation < mappings[index+1].activation) {
               TestMapping temp =  mappings[index];
               mappings[index] = mappings[index+1];
               mappings[index+1] = temp;
               ordered = false;
            }
        }
    } while (!ordered);
    
    printMap(mappings, words[0]);
    
    std::string testWord = words[0];
    std::string firstImage = removeWordType(mappings[0].imageWord);
    
    TImageFeatures wordImage = wordMap[testWord];
    if (testWord.compare(firstImage)==0) {
        
        if (wordImage.lastCorrect == TLT_CORRECT) {
            correctCorrect++;            
        } else if (wordImage.lastCorrect == TLT_INCORRECT) {
            incorrectCorrect++;
        } else {
            firstGuess++;
            dbgOut(1) << "First guess:\t" << testWord << endl;
        }
        
        wordImage.lastCorrect = TLT_CORRECT;
    } else {
        wordImage.lastCorrect = TLT_INCORRECT;
    }
    totalTest++;
    
    wordMap[testWord] = wordImage;
    
    dbgOut(1) << objectMap->size() << "\t";
    
    return true;
}

bool CrossSituationalTest::test(const std::string &line) {
    std::vector<std::string> words = split(line, '\t');
    
    MatVector<float> audioActivation = wordMap[words[0]].audioActivation;
    
    dbgOut(2) << words[0] << ": ";
    
    std::vector<TestMapping> mappings;        
    for (int i=1; i<words.size(); i++) {
        std::string word = wordMap[words[i]].word;
        
        MatVector<float> input = wordMap[words[i]].visualActivation;
        
        input.concat(audioActivation);
        
        if (isContext) {        //Context info
                context->train(input);
                input.concat(getContext());
        }
        
        GDSSOMMW::TNode *winner = objectMap->getWinner(input);
        
        TestMapping mapping;
        mapping.testWord = words[0];
        mapping.imageName = word;
        mapping.imageWord = words[i];
        mapping.nodeIndex = winner->getId();
        computeActivations(winner, mapping, input);    
        
        mappings.push_back(mapping);
        
        if (interestWord.compare(removeWordType(word))==0) {

            dbgOut(3) << "word: " << words[0] << endl;
            dbgOut(3) << "input :" << input.toString() << endl;
            dbgOut(3) << "node-" << mapping.nodeIndex << ":" << winner->w.toString() << endl;
        }
    }
    
    //Sord winner images
    bool ordered;
    do {
        ordered = true;
        for (int index=0; index<mappings.size()-1;index++) {
            if (mappings[index].activation < mappings[index+1].activation) {
               TestMapping temp =  mappings[index];
               mappings[index] = mappings[index+1];
               mappings[index+1] = temp;
               ordered = false;
            }
        }
    } while (!ordered);
    
    printMap(mappings,words[0]);
    
    std::string testWord = words[0];
    std::string firstImage = removeWordType(mappings[0].imageWord);
    std::string secondImage = removeWordType(mappings[1].imageWord);
    
    switch (testInstance) {
        case TI_SINGLE:
            nSingle++;
            if (testWord.compare(firstImage)==0)
                singleCorrect++;
            break;
        case TI_DOUBLE:
            nDouble++;
            if (testWord.compare(firstImage)==0) {
                if (testWord.compare(secondImage)==0) {
                    bothCorrect++;
                    if (endsWidth(mappings[0].imageWord,"-1")) {
                        firstCorrect++;
                    }
                } else
                    eitherCorrect++;
            } else
                if (testWord.compare(secondImage)==0) {
                    if (testWord.compare(firstImage)==0) {
                        bothCorrect++;
                        if (endsWidth(mappings[0].imageWord,"-1")) {
                                firstCorrect++;
                        }
                    } else
                        eitherCorrect++;
                }
            break;            
    }
    
    if (isInterestWord(testWord)) {
        totalContext++;
        if (testWord.compare(mappings[0].imageWord)==0) {
            correctWordContext++;
        }
    }
    
    return true;
}

CrossSituationalTest::~CrossSituationalTest() {

}

bool CrossSituationalTest::translateString(std::string text, MatMatrix<float> &audioInput) {
    std::string phonemes;
    FeaturesVector features;
    
    ttp.translateWords(text, phonemes);
    pf.translatePhonemesFeatures(phonemes, features);
    
    int inputCount = audioCB->getInputSize()/features.rows();    
    MatVector<float> vect(inputCount*features.rows());

    for (int j=0; j<inputCount + features.cols(); j++) {
        int start = max((int)inputCount - j - 1, 0);
        int end = min((int)inputCount-1-j+(int)features.cols(), (int)inputCount-1); 

        for (int k = 0; k<inputCount; k++) {
            for (int r=0; r<features.rows();r++) {
                int l = k*features.rows() + r;
                if (k>=start && k<=end) {
                    int index = j-inputCount+k;
                    vect[l] = features[r][index];
                } else
                    vect[l] = 0;
            }
        }

        audioInput.concatRows(vect);
    }
    
    //ArffData::rescaleCols01(audioInput);
    audioInput.add(1).mult(1.0/2);
}

void CrossSituationalTest::getAudioActivation(MatMatrix<float> &audioData, MatVector<float> &activationAudio) {
    
    for (int i=0; i<audioData.rows(); i++) {
        MatVector<float> row;
        MatVector<float> activation;
        
        audioData.getRow(i, row);          
        
        audioCB->getActivationVector(row, activation);
        
        float mean = activation.mean();
        for (int k=0; k<activation.size(); k++) {
            if (activation[k]>=mean)
                activation[k] = 1;
            else
                activation[k] = 0;
        }
        
        if (i == 0) {
            activationAudio = activation;
        } else {
            activationAudio += activation;
        }
    }
    
    if (normalizeInputs)
        activationAudio = activationAudio/activationAudio.norm();
    else
        activationAudio = activationAudio/audioData.rows();
}

bool CrossSituationalTest::getImageInput(std::string word, MatMatrix<float> &visualInput) {
    
    std::map<int, int> groupLabels;
    std::vector<int> groups;
    std::string filename = imagePath+word+".arff";
    if (ArffData::readArff(filename, visualInput, groupLabels, groups)) {
        //ArffData::rescaleCols01(visualInput);
        visualInput.mult(1.0/255);
        return true;
    };
    
    return false;
}

void CrossSituationalTest::getImageActivation(MatMatrix<float> &visualInput, MatVector<float> &activationVisual) {
    
        for (int i=0; i<visualInput.rows(); i++) {
        MatVector<float> row;
        MatVector<float> activation;
        
        visualInput.getRow(i, row);        
        
        visualCB->getActivationVector(row, activation);
                
        float mean = activation.mean();
        for (int k=0; k<activation.size(); k++) {
            if (activation[k]>=mean)
                activation[k] = 1;
            else
                activation[k] = 0;
        }
       
        if (i == 0) {
            activationVisual = activation;
        } else {
            activationVisual += activation;
        }
    }
        
    if (normalizeInputs)
        activationVisual = activationVisual/activationVisual.norm();
    else
        activationVisual = activationVisual/visualInput.rows();
    
}

void CrossSituationalTest::computeActivations(GDSSOMMW::TNode *winner, TestMapping &mapping, const MatVector<float> &input) {
    mapping.activation = objectMap->activation(*winner, input);
    
    int i;
    float distance = 0;
    float ds = 0;
    for (i=0; i<v_size; i++) {
        distance += winner->ds[i] * qrt((input[i] - winner->w[i]));
        ds+=winner->ds[i];
    }
    mapping.v_act = (ds / (ds + distance + 0.0000001));
    mapping.v_ds = ds/v_size;
    
    distance = 0;
    ds = 0;
    for (; i<v_size+a_size; i++) {
        distance += winner->ds[i] * qrt((input[i] - winner->w[i]));
        ds+=winner->ds[i];
    }
    mapping.a_act = (ds / (ds + distance + 0.0000001));
    mapping.a_ds = ds/a_size;
    
    distance = 0;
    ds = 0;
    for (; i<v_size+a_size+c_size; i++) {
        distance += winner->ds[i] * qrt((input[i] - winner->w[i]));
        ds+=winner->ds[i];
    }
    mapping.c_act = (ds / (ds + distance + 0.0000001));
    mapping.c_ds = ds/c_size;
    
}

void CrossSituationalTest::printDebug(GDSSOMMW::TNode *winner, const MatVector<float> &input, std::string &word, std::string &img) {
   
    float activation = objectMap->activation(*winner, input);
    dbgOut(0) << winner->getId() << ":" << word << "<>" << img << ":" <<activation << endl;
    
    float vam = 0, vdsm=0;
    float vdif = 0;
    int i=0;
    for (; i<111;i++) {
        vam += winner->a[i];
        vdsm += winner->ds[i];
        vdif += winner->ds[i] * qrt((input[i] - winner->w[i]));
    }
    float vac = (vdsm / (vdsm + vdif + 0.0000001));
    dbgOut(0) << "type\t\t" << "am" << "\t\t" << "dsm" << "\t\t" << "vdif" << "\t\t" << "ac" << endl;
    dbgOut(0) << "vis:\t\t" << vam << "\t\t" << vdsm << "\t\t" << vdif << "\t\t" << vac << endl;
    
    float aam = 0, adsm=0;
    float adif = 0;
    for (; i<input.size();i++) {
        aam += winner->a[i];
        adsm += winner->ds[i];
        adif += winner->ds[i] * qrt((input[i] - winner->w[i]));
    }
    float aac = (adsm / (adsm + adif + 0.0000001));
    dbgOut(0) << "aud:\t\t" << aam << "\t\t" << adsm << "\t\t" << adif << "\t\t" << aac << endl << endl;
}

bool CrossSituationalTest::isInterestWord(const std::string &word) {
    if (interestWord.compare("*")!=0)
        if (interestWord.compare(removeWordType(word))!=0)
            return false;
    
    return true;
}

void CrossSituationalTest::printMap(const std::vector<TestMapping> &mappings, const std::string &trainingWord) {
   
    if (!isInterestWord(trainingWord))
            return;
    
    TlastTest ll = wordMap[trainingWord].lastCorrect;
    dbgOut(1) << endl << fixsize(trainingWord,8);
    
    switch (ll) {
        
        case TLT_NONE:
            dbgOut(1) << "-N:";
            break;
        case TLT_CORRECT:
            dbgOut(1) << "-C:";
            break;
        case TLT_INCORRECT:
            dbgOut(1) << "-I:";
            break;
    }
        
    dbgOut(1) << std::fixed;
    dbgOut(1) << std::setprecision(5);
    
    dbgOut(1) << "\tCtxt\ta\tv_act\ta_act\tc_act\tv_ds\ta_ds\tc_ds" << endl;
    
    for (int i=0; i<mappings.size();i++) {
        TestMapping map = mappings[i];
        
        dbgOut(1) << map.nodeIndex << "-" << fixsize(map.imageWord,8) << "\tJ-" << context->J << ":\t" << map.activation;
        dbgOut(1) << "\t" << map.v_act << "\t" << map.a_act << "\t" << map.c_act;
        dbgOut(1) << "\t" << map.v_ds  << "\t" << map.a_ds  << "\t" << map.c_ds << endl;
    }
    dbgOut(1) << endl;
}