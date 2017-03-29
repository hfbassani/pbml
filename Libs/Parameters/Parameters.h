/**\mainpage Este projeto permite armazenar parâmetros em arquivos.
 * @author Hansenclever Bassani
 * 
 * As três classes principais são:
 * 
 * Parameters: define um conjunto de parâmetros
 * 
 * Parameter: define um parâmetro
 * 
 * CFGFile: define um arquivo onde os parâmetros serão armazenados.
 *
 * Exemplo de utilização:
 * \code
 * 
 * //Primeiramente é necessário definir uma classe derivada de Parameters que irá conter os parâmetros:
 *  
 * class MyParameters: public Parameters {
 * 
 * public:
 *      //Esta classe contém três parâmetros:
 *      Parameter<double> p1; //parametro double
 *      Parameter<float> p2;  //parametro float
 *      Parameter<int> p3;    //parametro inteiro
 * 
 *      //Em seguida definimos um construtor da seguinte maneira:
 *      MyParameters() {
 *          //Um nome para este cojunto de parâmetros.
 *          section = "Meus Parametros";
 * 
 *          //Uma descrição mais detalhada para este conjunto de parâmetros.
 *          comments = "Parametros do meu sistema p1 p2 e p3";
 * 
 *          //Indica quais parâmetros serão salvos em arquivo e um comentário 
 *          //explicativo para cada parâmetro que será incluído no arquivo.
 *          addParameterD(p1, "Este parametro controla 1");
 *          addParameterD(p2, "Este parametro controla 2");
 *          addParameterD(p3, "Este parametro controla 3");
 * 
 *          //Define valores padrão para os parâmetros
 *          p1 = 10.0;
 *          p2 = 100.;
 *          p3 = 1000;
 *      }
 * };
 * 
 *  //Agora podemos utilizar nossos parâmetros e savá-los em arquivo:
 *  int main(void)
 *  {
 *      //Instancia um arquivo de configuração
 *      CFGFile cfgFile("test.cfg");
 * 
 *      //Instancia um objeto do tipo MyParameters
 *      MyParameters params;
 * 
 *      //Se o arquivo já existe
 *      if (cfgFile.exists())
 *          cfgFile >> params; //Le os valores atuais
 *      else 
 *          cfgFile << params; //Caso contrário, grava um com os valores padrão
 * 
 *      //Imprime o arquivo na tela
 *      cout << params;
 * 
 *      //Imprime apenas alguns parâmetros
 *      cout << params.p1.name << ": " << params.p1 << "\t" << params.p2.name << ": " << params.p2 << endl;
 * 
 *      //Le um valor para um parâmetro
 *      cout << "Digite um novo valor para p1:";
 *      cin >> params.p1;
 *      cout << "Agora o valor de p1 é:" << params.p1 << endl;
 * 
 *      //Modifica o valor de outro parâmetro
 *      params.p2 = params.p2 + 20;
 *      cout << "Agora o valor de p2 é:" << params.p2 << endl;
 * 
 *      //Limpa o arquivo e grava os novos parâmetros
 *      cfgFile.erase() << params;
 * }
 * \endcode
 */
 
#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <list>
#include <map>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <typeinfo>

class Parameters;
template <class T> class Parameter;

/**
 * Classe base que representa uma seção de parâmetros em arquivo que armazena parâmetros.
 * @author Hansenclever Bassani
 */
class CFGFileObject {

    protected:
    /**     
     * Nome da sessão dos parâmetros no arquivo.
     */
    std::string section;

    /**     
     * Comentários de começo de arquivo.
     */
    std::string comments;

    /**     
     * Método virtual que deve converter esta classe para uma stream de parâmetros em texto.
     * 
     * @param out   Uma stream de saída.
     * @return      A própria stream recebida como entrada.
     */
    virtual std::ostream& toStream(std::ostream& out) = 0;
    
    /**     
     * Método virtual que deve carregar esta classe de uma stream de parâmetros em texto.
     * @param in    Uma stream de entrada.
     * @return      A própria stream recebida como entrada.
     */
    virtual std::istream& fromStream(std::istream& in) = 0;

    /**     
     * Operador binário que converte um CFGFileObject em uma stream.
     * @param out             Uma stream de saída.
     * @param cfgFileObject   Um objeto CFGFileObject.
     * @return                A própria stream recebida como entrada.
     */
    friend std::ostream& operator << (std::ostream& out, CFGFileObject &cfgFileObject);
    
    /**     
     * Operador binário que converte uma stream em um CFGFileObject.
     * @param in              Uma stream de entrada.
     * @param cfgFileObject   Um objeto CFGFileObject.
     * @return                A própria stream recebida.
     */
    friend std::istream& operator >> (std::istream& in, CFGFileObject &cfgFileObject);

    /**     
     * Encontra o início de seção em uma stream.
     * @param in              Uma stream de entrada.
     * @return                Verdadeiro se encontrou o início de seção.
     */    
    bool findSectionStart(std::istream& in);
    
    /**     
     * Lê a seção de comentários
     * @param in              Uma stream de entrada.
     * @return                Verdadeiro se encontrou os comentários.
     */       
    bool readSectionComments(std::istream& in);
};

/**     
 * Operador binário que converte um CFGFileObject em uma stream.
 * @param out             Uma stream de saída.
 * @param cfgFileObject   Um objeto CFGFileObject.
 * @return                A própria stream recebida como entrada.
 */
std::ostream& operator << (std::ostream& out, CFGFileObject &cfgFileObject);

/**     
 * Operador binário que converte uma stream em um CFGFileObject.
 * @param in              Uma stream de entrada.
 * @param cfgFileObject   Um objeto CFGFileObject.
 * @return                A própria stream recebida.
 */
std::istream& operator >> (std::istream& in, CFGFileObject &cfgFileObject);

/**     
 * Procura uma seção em uma string.
 * @param line            A string de entrada.
 * @param section         A seção encontrada, caso haja uma.
 * @return                Verdadeiro se encontrou uma seção.
 */   
bool getSectionFromLine(std::string const line, std::string &section);

/**     
 * Procura um id de parâmetro em uma string
 * @param line          A string de entrada.
 * @param id            Saída: id do parâmetro encontrado, caso haja um.
 * @return              Verdadeiro se encontrou uma parâmetro.
 */   
bool getIdFromLine(std::string const line, std::string &id);

/**     
 * Le o valor de um parâmetro em uma string
 * @param line          A string de entrada.
 * @param value         Saída: valor do parâmetro encontrado, caso haja um.
 * @return              Verdadeiro se encontrou o valor do parâmetro.
 */   
bool getValueFromLine(std::string const line, std::string &value);

/**     
 * Le a descrição de um parâmetro em uma string
 * @param line          A string de entrada.
 * @param description   Saída: a descrição do parâmetro encontrado, caso haja um.
 * @return              Verdadeiro se encontrou uma descrição de parâmetro.
 */
bool getDescriptionFromLine(std::string const line, std::string &description);

/**     
 * Verifica se a string contem um final de seção
 * @param line          A string de entrada.
 * @return              Verdadeiro se encontrou um final de seção.
 */
bool checkEndSection(std::string const line);


/**     
 * Classe base para um parâmetro
 * @author Hansenclever Bassani
 */
class ParameterBase {

public:
        /**     
         * Nome do parâmetro.
         */
	std::string name;
        
        /**     
         * Valor do parâmetro.
         */
	std::string description;

        /**     
         * Método virtual nulo que converte o parâmetro para uma string.
         * @return converte este parâmetro para uma representação de string.
         */
	virtual std::string toString() const = 0;
        
        /**     
         * Método virtual nulo que lê um parâmetro a partir de uma string.
         * @param A string contendo um parâmetro
         */        
	virtual void fromString(std::string& line) = 0;
};

/**     
 * Classe principal de parâmetros. 
 * Esta classe representa um conjunto de parâmetros a ser armazenado em arquivo.
 * Ela deve ser extendida e incluir parâmetros do tipo "Parameter" dos tipos desejados.
 * Estes parâmetros devem ser inicializados no construtor da classe.
 * No construtor também devem ser indicados os parâmetros que devem ser armazenados em arquivo
 * através das funções "addParameter"
 * @author Hansenclever Bassani
 */
class Parameters:public CFGFileObject {
public:

	//Declarações de tipos
	typedef std::list<ParameterBase *> ParametersList;
	typedef std::map<std::string, ParameterBase *> ParametersMap;

	//Listas e mapas
	ParametersList persistentParams;
	ParametersMap paramMap;

	/**     
	 * Construtor padrão
	 */
	Parameters() {
		section = typeid(this).name();
	}

	/**     
	 * Construtor que carrega parâmetros de um arquivo
	 */
	Parameters(const std::string fileName) {
		load(fileName);
	}

	/**     
	 * Construtor que seta a sessão e carrega parâmetros de um arquivo
	 */
	Parameters(const std::string fileName, const std::string section) {
		this->section = section;
		load(fileName);
	}

	/**     
	 * Adiciona à lista um parâmetro sem informar sua descrição
	 * @param parameter Parametro
	 */
	void addParameterN(ParameterBase &parameter, std::string name);

	/**     
	 * Adiciona à lista um parâmetro setando seu nome e descrição
	 * @param parameter Parametro
	 * @param name Nome
	 * @param description Descrição
	 */
	void addParameterND(ParameterBase &parameter, std::string name, std::string description);
        
	/**     
	 * Adiciona à lista um parâmetro setando seu nome e descrição
	 * @param parameter Parametro
	 * @param name Nome
	 * @param description Descrição
         * @param value Valor padrão
	 */
        template <class T> 
        void addParameterNDV(Parameter<T> &parameter, std::string name, std::string description, T value) {
            parameter.name = name;
            parameter.description = description;
            parameter.value = value;
            persistentParams.push_back(&parameter);
            paramMap[name] = &parameter;
        }


	/**     
	 *Esta macro pode ser chamada dentro de classes derivadas de ParametersBase
	 *Com objetivo de adicionar um parâmetro à lista de parametros com nome identico ao do identificador
	 * passado no primeiro parâmetro
	 * @param parameter Parametro
	 */
	#define addParameter(parameter) addParameterN(parameter, #parameter)

	/**     
	 *Esta macro pode ser chamada dentro de classes derivadas de ParametersBase
	 *Com objetivo de adicionar um parâmetro à lista de parametros com nome identico ao do identificador
	 * passado no primeiro parâmetro
	 * @param parameter Parametro
	 * @param description Descrição
	 */
	#define addParameterD(parameter, description) addParameterND(parameter, #parameter, description)

	/**     
	 *Esta macro pode ser chamada dentro de classes derivadas de ParametersBase
	 *Com objetivo de adicionar um parâmetro à lista de parametros com nome identico ao do identificador
	 * passado no primeiro parâmetro
	 * @param parameter Parametro
	 * @param description Descrição
         * @param value Valor padrão
	 */
	#define addParameterDV(parameter, description, value) addParameterNDV(parameter, #parameter, description, value)

	/**     
	 * Salva os parâmetros em um arquivo
	 * @param fileName Nome do arquivo
	 */
	void save(std::string fileName) {
		std::ofstream outfile (fileName.c_str());
		outfile << (*this);
		outfile.close();
	}

	/**     
	 * Lê os parâmetros de um arquivo
	 * @param fileName Nome do arquivo
	 */
	void load(std::string fileName) {
		std::ifstream infile;
		infile.open(fileName.c_str(), std::ifstream::in);
		if (infile.good())
			infile >> (*this);
		else
			std::cerr << "could not open the file:" << fileName << std::endl;
		infile.close();
	}


    std::ostream& toStream(std::ostream& out);
    std::istream& fromStream(std::istream& in);

};

/**     
 * Class que representa um arquivo de configuração.
 * @author Hansenclever Bassani
 */
class CFGFile
{
private:
	std::string fileName;
	std::ofstream fout;
	std::ifstream fin;

public:

	//Configurações de formatação do arquivo
	static const std::string delimiterStr;  // separator between key and value
	static const std::string commentStr;    // separator between value and comments
	static const std::string endValueStr;   // separator between value and comments
	static const std::string sectionStartDel; // string that defines the first string of section start
	static const std::string sectionEndDel;   // string that defines the last string of section start
	static const std::string sectionEnd;      // string that defines the section end delimiter

        /**     
         * Construtor
         * @param   nome do arquivo onde os parâmetros de serão armazenados
         * 
         */
	CFGFile(std::string fileName) {
		this->fileName = fileName;
	}

        /**     
         * Muda o nome do arquivo.
         * @param   nome do arquivo onde os parâmetros de serão armazenados.
         * 
         */
	CFGFile& setFileName(std::string fileName) {
            this->fileName = fileName;
            return *this;
	}
        
        /**     
	 * Verifica se o arquivo existe
         * @return verdadeiro se o arquivo existe.
	 */
        bool exists() {
            if (FILE * file = fopen(fileName.c_str(), "r")) {
                fclose(file);
                return true;
            } else {
                return false;
            }   
        }
        
	/**     
	 * Limpa arquivo de configuração
	 */
	CFGFile& erase()
	{
		fout.open(fileName.c_str());
		fout << "";
		fout.close();
		return *this;
	}
        
	/**     
	 * Grava seção de parâmetros em um arquivo de configuração.
	 */
	CFGFile& operator << (CFGFileObject& is) {
		fout.open(fileName.c_str(), std::ofstream::app);
		fout << is;
		fout.close();
		return *this;
	}

	/**     
	 * Lê parâmetros de um seção de um arquivo de configuração.
	 */
	CFGFile& operator >> (CFGFileObject& os) {

		fin.open(fileName.c_str());
		fin >> os;
		fin.close();
		return *this;
	}
};

/**     
 * Classe que representa um parâmetro de tipos básicos implementando os operadores da iostream =, << e >>
 * @author Hansenclever Bassani
 */
template <class T> class Parameter: public ParameterBase
{
public:
	T value;

	virtual inline Parameter<T>& operator=(const T& value)
	{
		this->value = value;
		return *this;
	}

	virtual inline operator T () const
	{
		return this->value;
	}

	virtual std::string toString() const {

		std::ostringstream ost;
		ost << name << "=" << value << CFGFile::endValueStr << CFGFile::commentStr << description;
		return ost.str();
	}

	virtual void fromString(std::string& str) {

		if (!getIdFromLine(str, name))
			std::cerr << "Error reading parameter name" << std::endl;

		std::string valueStr;
		if (getValueFromLine(str, valueStr))
		{
			std::istringstream ist(valueStr);
			ist >> value;
		}
		else
			std::cerr << "Error reading value for parameter" << name << std::endl;

		getDescriptionFromLine(str, description);
	}
        
        friend std::istream& operator >> (std::istream& in, Parameter<T>& p) {
            in >> p.value;
            return in;
        }

};

// trim from end
std::string &ltrim(std::string &s);
// trim from end
std::string &rtrim(std::string &s);
// trim from both ends
std::string &trim(std::string &s);
        
#endif /**     PARAMETERS_H_*/
