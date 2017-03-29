/* 
 * File:   StringHelper.h
 * Author: hans
 *
 * Created on 18 de Novembro de 2013, 09:08
 */

#ifndef STRINGHELPER_H
#define	STRINGHELPER_H

#include <string>
#include <vector>

///////////////// General use ///////////////////////
#ifndef __trim__
#define __trim__
// trim from start
std::string &ltrim(std::string &s);
// trim from end
std::string &rtrim(std::string &s);
// trim from both ends
std::string &trim(std::string &s);
#endif

// split string
std::vector<std::string> split(const std::string &s, char delim);
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);
bool endsWidth(std::string const &fullString, std::string const &ending);
// convert double to string
std::string toString(double x);
//fixed size str
std::string fixsize(const std::string& fname, int size);

/////////////////// File related rotines //////////////////
// remove file extension
std::string removeExtension(const std::string &filename);
// get directory of path
std::string dirnameOf(const std::string& fname);
// string ends with


#endif	/* STRINGHELPER_H */

