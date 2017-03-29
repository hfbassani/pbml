/* 
 * File:   ART2Context.cpp
 * Author: hans
 * 
 * Created on 27 de Outubro de 2009, 08:39
 */

#include "ART2Context.h"
#include "Defines.h"
#include <math.h>
#include <vector>

using namespace std;

ART2Context::ART2Context() {

}

ART2Context::~ART2Context() {
    /*cout << endl;
    cout << "P: " << &P << endl;
    cout << "Q: " << &Q << endl;
    cout << "R: " << &R << endl;
    cout << "S: " << &S << endl;
    cout << "U: " << &U << endl;
    cout << "V: " << &V << endl;
    cout << "W: " << &W << endl;
    cout << "X: " << &X << endl;
    cout << "Y: " << &Y << endl;
    cout << "UC: " << &UC << endl;
    cout << "PC: " << &PC << endl;*/
}

void ART2Context::initialiaze(CFGFile &parametersFile) {

   parametersFile >> parameters;

}

void ART2Context::saveParameters(CFGFile &parametersFile) {

    parametersFile << parameters;

}

float ART2Context::f(float x) {
    if (x>=parameters.theta)
        return x;
    else
        return 0;
}

void ART2Context::init(MatVector<float> &input) {
    init(input.size());
    
    for (int i=0; i<B.rows()/2;i++) {
        B[i][0] = input[i];
    }
    
    for (int i=0; i<B.rows()/2;i++) {
        B[i+B.rows()/2][0] = input[i];
    }
    
    J=0;
}

void ART2Context::init(int n) {
     //Alocate vectors and matrices
    parameters.n = n;
    P.size(parameters.n);
    Q.size(parameters.n);
    R.size(parameters.n);
    S.size(parameters.n);
    U.size(parameters.n);
    V.size(parameters.n);
    W.size(parameters.n);
    X.size(parameters.n);
    Y.size(1);
    UC.size(parameters.n); UC.fill(0);
    PC.size(2*parameters.n);
    parameters.theta = 1.0/sqrt((float)n);

    T.size(1, 2*parameters.n);
    B.size(2*parameters.n, 1);
    
    T.random();
    //T.fill(0.1);
    T.normalize();
    
    B.random();    
    //B.fill(0.1);
    B.normalize();
}

const MatVector<float> &ART2Context::getWinnerPrototype(bool denorm) {

    if (J>=0 && J<B.cols()) {
        int size = B.rows()/2;
        Prototype.size(size);
        for (int i=0; i<size;i++) {
            Prototype[i] = B[i][J];
        }
        
    }

    if (denorm && parameters.denorm_weight>0)
        Prototype.mult(Prototype.size()*parameters.denorm_weight);

    return Prototype;
}

const MatVector<float> &ART2Context::getCurrentContext(bool denorm) {
    
    if (J>=0 && J<B.cols()) {
        int size = B.rows()/2;
        Context.size(size);
        for (int i=0; i<size;i++) {
            Context[i] = B[i+size][J];
        }
        
    }
    
    if (denorm && parameters.denorm_weight>0)
        Context.mult(Context.size()*parameters.denorm_weight);
    
    return Context;
}

const MatVector<float> &ART2Context::getOutput(bool denorm) {
    
    B.getCol(J, Context);
    
    if (denorm && parameters.denorm_weight>0)
        Context.mult(Context.size()*parameters.denorm_weight);
    
    return Context;
}

void ART2Context::train(MatVector<float> &S)
{
    uint n = parameters.n;
    
    //Inicialize as ativações nas unidades de F1:
    float nS = S.norm(); //Pre-calcula norma de S
    for (int i=0; i<parameters.n; i++)
    {
        U[i] = 0; W[i] = S[i]; P[i] = 0; Q[i] = 0;
        X[i] = S[i] / (parameters.e + nS); V[i] = f(X[i]);
    }

    //Atualize as ativações nas unidades de F1 novamente:
    float nV = V.norm(); //Pre-computa as normas de V, W e P
    float nW = W.norm();
    float nP = P.norm();
    for (int i=0; i<parameters.n; i++)
    {
        U[i] = V[i] / ( parameters.e + nV );
        W[i] = S[i] + parameters.a*U[i];
        P[i] = U[i];
        X[i] = W[i] / ( parameters.e + nW );
        Q[i] = P[i] / ( parameters.e + nP );
        V[i] = f(X[i]) + parameters.b*f(Q[i]);
    }

    //Propague os valores para as unidades UC:
    for (int i=0; i<parameters.n; i++)
    {
        UC[i] = parameters.back * UC[i] + (1-parameters.back) * f(U[i]);
    }

    //Normalize unidades de contexto:
    float nUC = UC.norm(); //Pre-computa as normas de UC
    for (int i=0; i<parameters.n; i++)
    {
        UC[i] = UC[i] / ( parameters.e + nUC );
    }

    //Propague para unidades PC:
    for (int i=0; i<parameters.n; i++)
    {
        PC[i] = UC[i];
    }

    //Calcule os sinais para as unidades em F2:
    for (uint j=0; j<Y.size(); j++) // Para cada unidade J
    {
        float sumBijPi = 0;
        float sumBinjPCi = 0;
        for (int i=0; i<parameters.n; i++)  //Somatório em i
        {
                sumBijPi += B[i][j]*P[i];
                sumBinjPCi += B[i+n][j]*PC[i];
        }

        Y[j] = (1 - parameters.contextWeight)*sumBijPi + (parameters.contextWeight)*sumBinjPCi;
    }

    /*cout << "U: " << U.toString() << endl;
    cout << "W: " << W.toString() << endl;
    cout << "S: " << S.toString() << endl;
    cout << "P: " << P.toString() << endl;
    cout << "Q: " << P.toString() << endl;
    cout << "X: " << X.toString() << endl;
    cout << "V: " << V.toString() << endl;
    cout << "UC: " << UC.toString() << endl;
    cout << "PC: " << PC.toString() << endl;*/

    reset = true;

    MatVector<float> Y2 = Y;
    //Enquanto houver reset faça
    while (reset)
    {
        // Determine J: o nó j em F2 com maior ativação yj
        Y.max(J);
        
        // Se yJ = –1 então todos os nós estão inibidos e este padrão não pode ser
        // agrupado nos grupos existentes, somente num novo grupo:
        if (Y[J] == -1)
        {
            //Aloca mais espaço em Y, B e T

            Y.append(0);
            MatVector<float> v = getOutput(false);
            T.concatRows(v);
            B.concatCols(v);

            //J = uma unidade ainda não utilizada
            J = Y.size()-1;
            reset = false;
        }
        else
        {  // Verifique se houve reset:
            float nV = V.norm(); //Pre-computa as normas de V, U, P e PC
            for (int i=0; i<parameters.n; i++)
            {
                U[i] = V[i] / ( parameters.e + nV );
                P[i] = U[i] + parameters.d*T[J][i];
                PC[i+n] = T[J][i+n];
            }
            
            float nU = U.norm();
            float nP = P.norm();
            float nPC = PC.norm();
            
            for (int i=0; i<parameters.n; i++)
                R[i] = (U[i] + parameters.c*P[i] + parameters.contextWeight*PC[i])/(parameters.e + nU + parameters.c*nP + parameters.contextWeight*nPC);

            if (R.norm() < (float)(parameters.rho - parameters.e))
            {
                reset=true;
                Y[J] = -1; //(inibe unidade J)
            }
            else // Senão
            {
                reset = false;
                float nW = W.norm();  //Pre-computa as normas de W e P
                float nP = P.norm();
                for (int i=0; i<parameters.n; i++)
                {
                    W[i] = S[i] + parameters.a*U[i];
                    X[i] = W[i] / (parameters.e + nW);
                    Q[i] = P[i] / (parameters.e + nP);
                    V[i] = f(X[i]) + parameters.b*f(Q[i]);
                }
            }
        }

        //Se não houve reset repita nro_iterações vezes
        if (!reset)
        {
            for (int l=0; l< parameters.nbrIterations; l++)
            {
                //Atualize os pesos da unidade vencedora J
                for (int i=0; i<parameters.n; i++)
                {
                    T[J][i] = parameters.alpha*parameters.d*U[i] + (1 + parameters.alpha* parameters.d* (parameters.d* - 1))*T[J][i];
                    B[i][J] = parameters.alpha* parameters.d* U[i] + (1 + parameters.alpha* parameters.d* (parameters.d* - 1))*B[i][J];
                    T[J][i+n] = parameters.alpha_context* parameters.d_context* UC[i] + (1 + parameters.alpha_context* parameters.d_context* (parameters.d_context* - 1))*T[J][i+n];
                    B[i+n][J] = parameters.alpha_context* parameters.d_context* UC[i] + (1 + parameters.alpha_context* parameters.d_context* (parameters.d_context* - 1))*B[i+n][J];
                }

                //Normalizar vetores atualizados
                MatVector<float> TJ, BJ;
                T.getRow(J, TJ);
                B.getCol(J, BJ);
                float nTJ = TJ.norm(), nBJ = BJ.norm();
                for (int i=0; i<parameters.n; i++)
                {
                    T[J][i] = T[J][i] / nTJ;
                    B[i][J] = B[i][J] / nBJ;
                    T[J][i+n] = T[J][i+n] / nTJ;
                    B[i+n][J] = B[i+n][J] / nBJ;
                }

                //Atualize as ativações em F1
                float nV = V.norm(); //Pre-computa as normas de V, W e P
                float nW = W.norm();
                float nP = P.norm();
                for (int i=0; i<parameters.n; i++)
                {
                    U[i] = V[i] / ( parameters.e + nV);
                    W[i] = S[i] + parameters.a*U[i];
                    P[i] = U[i] + parameters.d* T[J][i];
                    X[i] = W[i] / ( parameters.e + nW);
                    Q[i] = P[i] / ( parameters.e + nP);
                    V[i] = f(X[i]) + parameters.b*f(Q[i]);
                }
            }
        }
    }
    
    dbgOut(0) << R.norm() << "\t" << J << "\t" << getCurrentContext(false).toCSV() << endl;
}

void ART2Context::train(MatMatrix<float> &Ss)
{
    if (Ss.cols() != (uint)parameters.n)
    {
        cerr << "Wrong number of input data. Found: " << Ss.cols() << " Expected: " << (int)parameters.n;
        return;
    }

    MatVector<float> S;

    //Alocate vectors and matrices
    P.size(parameters.n);
    Q.size(parameters.n);
    R.size(parameters.n);
    S.size(parameters.n);
    U.size(parameters.n);
    V.size(parameters.n);
    W.size(parameters.n);
    X.size(parameters.n);
    Y.size(1);
    UC.size(parameters.n);
    PC.size(2*parameters.n);

    T.size(1, 2*parameters.n);
    B.size(2*parameters.n, 1);

    //Inicialize as matrizes e vetores
    T.random();
    B.random();
    UC.fill(0);

    //Repita nro_epocas vezes
    for (int k=0; k<parameters.nbrEpochs; k++)
    {
        //Para cada vetor de entrada s faça
        for (uint nPattern=0; nPattern<Ss.rows(); nPattern++)
        {
            Ss.getRow(nPattern, S);
            cout << "Cos " << nPattern << ": " << cos(S, PC) << endl;
            train(S);
        }
    }
}

int ART2Context::recognize(MatVector<float> &S)
{
    uint n = parameters.n;
    float rho = parameters.rho;
    
    //Inicialize as ativações nas unidades de F1:
    float nS = S.norm(); //Pre-calcula norma de S
    for (int i=0; i<parameters.n; i++)
    {
        U[i] = 0; W[i] = S[i]; P[i] = 0; Q[i] = 0;
        X[i] = S[i] / (parameters.e + nS); V[i] = f(X[i]);
    }

    //Atualize as ativações nas unidades de F1 novamente:
    float nV = V.norm(); //Pre-computa as normas de V, W e P
    float nW = W.norm();
    float nP = P.norm();
    for (int i=0; i<parameters.n; i++)
    {
        U[i] = V[i] / ( parameters.e + nV );
        W[i] = S[i] + parameters.a*U[i];
        P[i] = U[i];
        X[i] = W[i] / ( parameters.e + nW );
        Q[i] = P[i] / ( parameters.e + nP );
        V[i] = f(X[i]) + parameters.b*f(Q[i]);
    }

    //Propague os valores para as unidades UC:
    for (int i=0; i<parameters.n; i++)
    {
        UC[i] = parameters.back * UC[i] + (1-parameters.back) * f(U[i]);
    }

    //Normalize unidades de contexto:
    float nUC = UC.norm(); //Pre-computa as normas de UC
    for (int i=0; i<parameters.n; i++)
    {
        UC[i] = UC[i] / ( parameters.e + nUC );
    }

    //Propague para unidades PC:
    for (int i=0; i<parameters.n; i++)
    {
        PC[i] = UC[i];
    }

    //Enquanto não houver encontrado unidade suficientemente semelhante em F2 faça
    do
    {
        //Habilite todas as unidades em F2 e calcule os sinais para as unidades em F2:
        for (int i=0; i<parameters.n; i++)
        {
            for (uint j=0; j<Y.size(); j++) // Para cada unidade J
            {
                float sumBijPi = 0;
                float sumBinjPCi = 0;
                for (int i=0; i<parameters.n; i++)  //Somatório em i
                {
                        sumBijPi += B[i][j]*P[i];
                        sumBinjPCi += B[i+n][j]*PC[i];
                }

                Y[j] = (1 - parameters.contextWeight)*sumBijPi + (parameters.contextWeight)*sumBinjPCi;
            }
        }
        reset = true;

        //Enquanto houver reset e houver unidades habilitadas em F2 faça
        bool enabled = true;
        while (reset && enabled)
        {
            // Determine J: o nó j em F2 com maior ativação yj
            Y.max(J);

            // Verifique se houve reset:
            float nV = V.norm(); //Pre-computa as normas de V, U, P e PC
            for (int i=0; i<parameters.n; i++)
            {
                U[i] = V[i] / ( parameters.e + nV );
                P[i] = U[i] + parameters.d*T[J][i];
                PC[i+n] = T[J][i+n];
            }
            
            float nU = U.norm();
            float nP = P.norm();
            float nPC = PC.norm();
            
            for (int i=0; i<parameters.n; i++)
                R[i] = (U[i] + parameters.c*P[i] + parameters.contextWeight*PC[i])/(parameters.e + nU + parameters.c*nP + parameters.contextWeight*nPC);

            if (R.norm() < (float)(rho - parameters.e))
            {
                reset=true;
                Y[J] = -1; //(inibe unidade J)
            }
            else
            {  // Senão
                reset = false;
            }

            //Verifica se se alguma unidades está habilitada:
            enabled = false;
            for (uint j=0; j<Y.size(); j++)
                if (Y[j] != -1)
                  enabled = true;
        }

        if (!enabled) //Se não houver unidade habilitada em F2 então:
        {
            rho = rho*0.999; //reduz a similaridade exigida
        }
        else
        {
            return J;
        }
    }
    while (true);
}

void ART2Context::recognize(MatMatrix<float> &Ss, MatVector<int> &ry)
{
    if (Ss.cols() != (uint)parameters.n)
    {
        cerr << "Wrong number of input data. Found: " << Ss.cols() << " Expected: " << (int)parameters.n;
        return;
    }

    ry.size(Ss.rows());
    ry.fill(-1);

    MatVector<float> S;

    //Repita nro_epocas vezes
    for (int k=0; k<parameters.nbrEpochs; k++)
    {
        //Para cada vetor de entrada s faça
        for (uint nPattern=0; nPattern<Ss.rows(); nPattern++)
        {
            Ss.getRow(nPattern, S);
            ry[nPattern] = recognize(S);
        }
    }
}

float ART2Context::cos(MatVector<float> &x, MatVector<float> &y)
{
    uint size = x.size();
    float sumx2 = 0, sumy2 = 0, sumxy = 0;

    for (uint i=0; i<size; i++)
    {
        sumx2 += x[i]*x[i];
        sumy2 += y[i]*y[i];
        sumxy += x[i]*y[i];
    }

    if (sumx2*sumy2==0) return 0;
    
    return (sumxy/(sqrt(sumx2)*sqrt(sumy2)));
}

void ART2Context::printClusters(MatVector<int> &ry, const std::string &wordsTestFileName)
{
    uint numClusters = Y.size();
    cout << numClusters << " unidades criadas: " << endl;

    string words[ry.size()];
    ifstream wordsTestFile(wordsTestFileName.c_str());

    for (uint w = 0; w < ry.size(); w++)
    {
        if (wordsTestFile.eof())
        {
            cerr << "Wrong number of words in file." << endl;
        }
        //Read a word from wordsList file
        wordsTestFile >> words[w];
    }

    for (uint c = 0; c < numClusters; c++)
    {
        cout << "Cluster "<< c << ":";
        for (uint w = 0; w < ry.size(); w++)
        {
            if (ry[w]==(int)c)
                cout << " " << words[w];
        }
        cout << endl;
    }
}