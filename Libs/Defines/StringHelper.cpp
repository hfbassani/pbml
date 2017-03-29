/* 
 * File:   StringHelper.cpp
 * Author: hans
 * 
 * Created on 18 de Novembro de 2013, 09:08
 */

#include "StringHelper.h"
#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>
#include <sstream>

///////////////// General use ///////////////////////
#ifndef __trim__
#define __trim__
// trim from start
std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}

// trim from end
std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}

// trim from both ends
std::string &trim(std::string &s) {
        return ltrim(rtrim(s));
}
#endif

//split
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

//bool
bool endsWidth(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

//toString
std::string toString(double x)
{
  std::ostringstream o;
  o << x;
  return o.str();
}

//fixed size str
std::string fixsize(const std::string& str, int size) {
    if (str.size()>size)
        return str.substr(0, size);
    else {
        std::string newstr = str;
        while (newstr.size() < size) {
            newstr = newstr + " ";
        }
        
        return newstr;
    }
}

/////////////////// File related rotines //////////////////

// remove file extension
std::string removeExtension(const std::string &filename) {
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot); 
}

// get directory of path
std::string dirnameOf(const std::string& fname)
{
     size_t pos = fname.find_last_of("\\/");
     return (std::string::npos == pos)
         ? ""
         : fname.substr(0, pos);
}