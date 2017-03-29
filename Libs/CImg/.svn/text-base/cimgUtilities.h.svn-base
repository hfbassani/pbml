/* 
 * File:   cimgUtilities.h
 * Author: daniel
 *
 * Created on 22 de Março de 2009, 19:19
 */

#ifndef _CIMGUTILITIES_H
#define	_CIMGUTILITIES_H
#include <map>
#include <set>
#include <iostream>
#include <sstream>
#include <string>
#include <CImg.h>
#include <MatVector.h>
#include <MatVectorList.h>
#include <MatFunctions.h>
#include <dirUtilities.h>
#include <getch.h>


typedef std::list<MatVector<float> > TVectorList;
const int normalOrdem[]={0,1,2};

void img2threshold(cimg_library::CImg<unsigned char> &img,int limiar, int x0=0, int y0=0, int x1=0, int y1=0);
void img2outline(cimg_library::CImg<unsigned char> &img,const int maskSize=3);
void img2colorList(const cimg_library::CImg<unsigned char> &img, MatVectorList<float> &vList, bool randomSamples=false, int samplesCount=1000,const int index[]=normalOrdem);
void getImgStatistics(const cimg_library::CImg<unsigned char> &img, MatVector<float> &vstat);
void getImgHistogram(const cimg_library::CImg<unsigned char> &img, MatVector<float> &v, bool imgMonocrome = false);
void getImgHistogram(const cimg_library::CImg<unsigned char> &img, MatVector<float> &v, int N, bool imgMonocrome = false);
void getImgMoms(const cimg_library::CImg<unsigned char> &img, MatVector<float> &vmom,int color,bool mono=false);
void img2pointList(const cimg_library::CImg<unsigned char> &img, TVectorList &vList ,int limiar, float z, float ax=1.0f, float ay=1.0f, float bx=0.0f,float by=0.0f);
void img2pointList(const cimg_library::CImg<unsigned char> &img, TVectorList &vList ,int limiar, float ax=1.0f, float ay=1.0f, float bx=0.0f,float by=0.0f);
void scannerImg(const cimg_library::CImg<unsigned char> &img,int xL0,int yL0, int xL1, int yL1,void f(int x,int y,const cimg_library::CImg<unsigned char> &imgWindow), int xLstep=1, int yLstep=1, bool borderCondition=true);
void pointList2wrl(std::ofstream &outputFile,TVectorList &vlist);
void img2wrl(std::ofstream &outputfile,cimg_library::CImg<unsigned char> &img,int limiar);
void imgList2wrl(std::ofstream &outputfile,TNameVector names,int limiar,float scale);
template<class T1,class T2> void img2Matrix(const cimg_library::CImg<T1> &img,MatMatrix<T2> &m,int channel);
unsigned char** makeColorPallet(const int &colors);
unsigned char** makeMonoPallet(const int colors, const int ch=0);
void destroyPallet(unsigned char** pallet);

template<class T> void saveImg(cimg_library::CImg<T> &img, std::string name, std::string label="", int number=0, std::string extName="png")
{
    using namespace std;
    
    stringstream out;
    
    out << name << label << number << "." << extName;
    
    img.save(out.str().c_str());
}

template<class T1, class T2> inline void roi2Vector(int x,int y,int direction, int s0, int s1, const cimg_library::CImg<T1> &img, MatVector<T2> &v) {
    //coloca os valores de s0 a s1(inclusive) em um vetor

    int L=(1 + s1 - s0);
    int shift=0;
    
    v.size(img.dimv()*L);

    if (direction == 0) {
        x+=s0;
        for (int ch = 0; ch < img.dimv(); ch++)
        {
            shift=ch*L;
            for (int i = 0; i < L; i++) {
                v[i+shift] = img.at(x+i,y,ch);
            }
        }
    } else
        if (direction == 1) {
            y-=s0;
        for (int ch = 0; ch < img.dimv(); ch++)
        {
            shift=ch*L;
            for (int i = 0; i < L; i++) {
                v[i+shift] = img.at(x,y-i,ch);
            }
        }
    } else
        if (direction == 2) {
            x-=s0;
        for (int ch = 0; ch < img.dimv(); ch++)
        {
            shift=ch*L;
            for (int i = 0; i < L; i++) {
                v[i+shift] = img.at(x-i,y,ch);
            }
        }
    } else {
            y+=s0;
        for (int ch = 0; ch < img.dimv(); ch++)
        {
            shift=ch*L;
            for (int i = 0; i < L; i++) {
                v[i+shift] = img.at(x,y+i,ch);
            }
        }
    }
};

template<class T1, class T2> inline void roi2Vector(float x0, float y0, float n, float s0, float s1, const cimg_library::CImg<T1> &img, MatVectorList<T2> &vList) {
    //coloca os valores de s0 a s1(inclusive) em um vetor
    int L = (1 + s1 - s0);
    int shift = 0;

    MatVector<float> v;
    v.size(img.dimv() * L);

//    using namespace cimg_library;
//    unsigned char black=0;
//    CImg<unsigned char> dispImg(img.width,img.height,1,1,255);
//    CImgDisplay disp;

    float x = 0, y = 0;
    float step = MatUtils::TWO_PI / n;
    for (float a = 0; a < MatUtils::TWO_PI; a += step) {
        x = x0 + s0 * cos(a);
        y = y0 + s0 * sin(a);
        v.zeros();

        for (float i = 0; i < L; i++) {
            x += cos(a);
            y += sin(a);
            for (int ch = 0; ch < img.dimv(); ch++) {
                shift = ch*L;
                v[i + shift] = img.at(x,y, ch);
//                dispImg.draw_point(x,y,&black);
            }
        }
        vList.push_back(v);
    }
//    if((x0>L)&&(y0>L))
//    {
//        disp.display(dispImg);
//        MatUtils::delay(0.1);
//    }
    //getch();
};

template<class T1, class T2> inline void roiHist2Vector(float x0, float y0, float n, float s0, float s1, const cimg_library::CImg<T1> &img, MatVectorList<T2> &vList) {
    //coloca os valores de s0 a s1(inclusive) em um vetor
    int L = (1 + s1 - s0);
    int shift = 0;

    MatVector<float> v;
    MatVector<float> vh;
    v.zeros(img.dimv() * L);

//    using namespace cimg_library;
//    unsigned char black=0;
//    CImg<unsigned char> dispImg(img.width,img.height,1,1,255);
//    CImgDisplay disp;

    float x = 0, y = 0;
    float step = MatUtils::TWO_PI / n;
    for (float a = 0; a < MatUtils::TWO_PI; a += step) {
        x = x0 + s0 * cos(a);
        y = y0 + s0 * sin(a);
        v.zeros();

        for (float i = 0; i < L; i++) {
            x += cos(a);
            y += sin(a);
            for (int ch = 0; ch < img.dimv(); ch++) {
                shift = ch*L;
                v[i + shift] = img.at(x,y, ch);
//                dispImg.draw_point(x,y,&black);
            }
        }
        hist(v,vh,0,255,5);
        vList.push_back(vh);
    }
//    if((x0>L)&&(y0>L))
//    {
//        disp.display(dispImg);
//        MatUtils::delay(0.1);
//    }
    //getch();
};

template<class T1, class T2> inline void roiHistRGB2Vector(float x0, float y0, float n, float s0, float s1, const cimg_library::CImg<T1> &img, MatVectorList<T2> &vList) {
    //coloca os valores de s0 a s1(inclusive) em um vetor
    int L = (1 + s1 - s0);
    int shift = 0;

    MatVector<float> vr,vg,vb;
    MatVector<float> vh,vhr,vhg,vhb;
    vr.zeros(L);
    vg.zeros(L);
    vb.zeros(L);

//    using namespace cimg_library;
//    unsigned char black=0;
//    CImg<unsigned char> dispImg(img.width,img.height,1,1,255);
//    CImgDisplay disp;

    float x = 0, y = 0;
    float step = MatUtils::TWO_PI / n;
    for (float a = 0; a < MatUtils::TWO_PI; a += step) {
        x = x0 + s0 * cos(a);
        y = y0 + s0 * sin(a);
        
        for (float i = 0; i < L; i++) {
            x += cos(a);
            y += sin(a);
            vr[i] = img.at(x,y,0,0);
            vg[i] = img.at(x,y,0,1);
            vb[i] = img.at(x,y,0,2);
//                dispImg.draw_point(x,y,&black);
        }
        hist(vr,vhr,0,255,L);
        hist(vg,vhg,0,255,L);
        hist(vb,vhb,0,255,L);
        vhr.concat(vhg,vhb);
        vList.push_back(vhr);
    }
//    if((x0>L)&&(y0>L))
//    {
//        disp.display(dispImg);
//        MatUtils::delay(0.1);
//    }
    //getch();
};

template<class T1, class T2> inline void roiMVSKRGB2Vector(float x0, float y0, float n, float s0, float s1, const cimg_library::CImg<T1> &img, MatVectorList<T2> &vList) {
    //coloca os valores de s0 a s1(inclusive) em um vetor
    int L = (1 + s1 - s0);
    int shift = 0;

    MatVector<float> vr,vg,vb;
    MatVector<float> vh,vhr,vhg,vhb;
    vr.zeros(L);
    vg.zeros(L);
    vb.zeros(L);

//    using namespace cimg_library;
//    unsigned char black=0;
//    CImg<unsigned char> dispImg(img.width,img.height,1,1,255);
//    CImgDisplay disp;

    float x = 0, y = 0;
    float step = MatUtils::TWO_PI / n;
    for (float a = 0; a < MatUtils::TWO_PI; a += step) {
        x = x0 + s0 * cos(a);
        y = y0 + s0 * sin(a);

        for (float i = 0; i < L; i++) {
            x += cos(a);
            y += sin(a);
            vr[i] = img.at(x,y,0,0);
            vg[i] = img.at(x,y,0,1);
            vb[i] = img.at(x,y,0,2);
//                dispImg.draw_point(x,y,&black);
        }
        getMVSK<float>(vr,vhr);
        getMVSK<float>(vg,vhg);
        getMVSK<float>(vb,vhb);
        vhr.concat(vhg,vhb);

        vList.push_back(vhr);
    }
//    if((x0>L)&&(y0>L))
//    {
//        disp.display(dispImg);
//        MatUtils::delay(0.1);
//    }
    //getch();
};

template<class TMatrix, class TImg> inline void copyM2Img(const TMatrix &m, TImg &f, int ch)
{
    int i,j;
    int imax,jmax;
    imax=MatUtils::min(m.rows(),f.dimx());
    jmax=MatUtils::min(m.cols(),f.dimy());
    for(i=0;i<imax;i++)
        for(j=0;j<jmax;j++)
        {
            f(i,j,ch)=m[i][j];
        }
};

template<class TMatrix, class TImg> inline void copyImg2M(const TImg &f,int ch, TMatrix &m)
{
    int i,j;
    int imax,jmax;
    imax=MIN(m.rows(),f.dimx());
    jmax=MIN(m.cols(),f.dimy());

    for(i=0;i<imax;i++)
        for(j=0;j<jmax;j++)
        {
            m[i][j]=f(i,j,ch);
        }
};

template<class T> float calcME(cimg_library::CImg<T> &testImg,const T *ftc,cimg_library::CImg<T> &groundTruth,const T *foc)
{
    float me=0;

    float foregroundCount=0;
    float backgroundCount=0;
    float foIft=0;
    float boIbt=0;

    for(int x=0;x<testImg.dimx();x++)
        for(int y=0;y<testImg.dimy();y++)
        {
            if((groundTruth(x,y,0)==foc[0])&&(groundTruth(x,y,1)==foc[1])&&(groundTruth(x,y,2)==foc[2]))
            {
                foregroundCount++;

                if((testImg(x,y,0)==ftc[0])&&(testImg(x,y,1)==ftc[1])&&(testImg(x,y,2)==ftc[2]))
                    foIft++;

            }
            else
            {
                backgroundCount++;

                if((testImg(x,y,0)!=ftc[0])||(testImg(x,y,1)!=ftc[1])||(testImg(x,y,2)!=ftc[2]))
                    boIbt++;
            }

        }
    me=1.0-(boIbt+foIft)/(backgroundCount+foregroundCount);
    return me;
};

template<class T> float calcME(const cimg_library::CImg<T> &testImg,const T &ftc,cimg_library::CImg<T> &groundTruth,const T &foc,int ch=0)
{
    if((testImg.dimx()!=groundTruth.dimx())||(testImg.dimy()!=groundTruth.dimy()))
    {
        errMsg("As imagens precisam ter a mesma dimensão!");
        return 1;
    }

    float me=0;
    float foregroundCount=0;
    float backgroundCount=0;
    float foIft=0;
    float boIbt=0;
    
    for(int x=0;x<testImg.dimx();x++)
        for(int y=0;y<testImg.dimy();y++)
        {
            if(groundTruth(x,y,ch)==foc)
            {
                foregroundCount++;

                if(testImg(x,y,ch)==ftc)
                    foIft++;

            }
            else
            {
                backgroundCount++;

                if(testImg(x,y,ch)!=ftc)
                    boIbt++;
            }

        }
    me=1.0-(boIbt+foIft)/(backgroundCount+foregroundCount);
    return me;
};

//template<class T> float calcNU(cimg_library::CImg<T> &testImg,const T *ftc)
//{
//    float nu=0;
//
//    float foregroundCount=0;
//    float backgroundCount=0;
//    float ftmx=0;
//    float ftmy=0;
//    float mx=testImg.dimx()/2;
//    float my=testImg.dimy()/2;
//    float varft;
//    float var;
//
//
//    for(int x=0;x<testImg.dimx();x++)
//        for(int y=0;y<testImg.dimy();y++)
//        {
//            if((testImg(x,y,0)==ftc[0])&&(testImg(x,y,1)==ftc[1])&&(testImg(x,y,2)==ftc[2]))
//            {
//                ftmx+=x;
//                ftmy+=y;
//                foregroundCount++;
//            }
//            else
//                backgroundCount++;
//        }
//
//    ftmx/=foregroundCount;
//    ftmy/=foregroundCount;
//
//    for(int x=0;x<testImg.dimx();x++)
//        for(int y=0;y<testImg.dimy();y++)
//        {
//            if((testImg(x,y,0)==ftc[0])&&(testImg(x,y,1)==ftc[1])&&(testImg(x,y,2)==ftc[2]))
//            {
//                varft+=((x-ftmx)*(x-ftmx)+(y-ftmy)*(y-ftmy));
//            }
//
//            var+=((x-mx)*(x-mx)+(y-my)*(y-my));
//        }
//
//    nu=(foregroundCount*varft)/((backgroundCount+foregroundCount)*var);
//    return nu;
//};

template<class T> float calcNU(const cimg_library::CImg<T> &testImg,const T *ftc,int ch=0)
{
    double nu=0;

    double foregroundCount=0;
    double backgroundCount=0;
    double ftmx=0;
    double ftmy=0;
    double mx=0.5;//testImg.dimx()/2;
    double my=0.5;//testImg.dimy()/2;
    double varft=0;
    double var=0;
    double xmaxval=testImg.dimx();
    double ymaxval=testImg.dimy();
    double N=testImg.dimx()*testImg.dimy();


    for(int x=0;x<testImg.dimx();x++)
        for(int y=0;y<testImg.dimy();y++)
        {
            if(testImg(x,y,ch)==ftc[ch])
            {
                ftmx+=x/xmaxval;
                ftmy+=y/ymaxval;
                foregroundCount++;
            }
            else
                backgroundCount++;
        }


    ftmx/=foregroundCount;
    ftmy/=foregroundCount;
    foregroundCount/=N;
    backgroundCount/=N;
    for(int x=0;x<testImg.dimx();x++)
        for(int y=0;y<testImg.dimy();y++)
        {
            double xx,yy;
            xx=x/xmaxval;
            yy=y/ymaxval;
            if(testImg(x,y,ch)==ftc[ch])
            {
                varft+=((xx-ftmx)*(xx-ftmx)+(yy-ftmy)*(yy-ftmy));
            }

            var+=((xx-mx)*(xx-mx)+(yy-my)*(yy-my));
        }

    nu=(foregroundCount*varft)/((backgroundCount+foregroundCount)*var);
    return nu;
};


template<class T> void sgtd(const cimg_library::CImg<T> &img, cimg_library::CImg<float> &imgStat, int xL = 1, int yL = 1, bool displayImg = false) {
    using namespace std;
    using namespace cimg_library;
//#define _debugSgtd
    #ifdef _debugSgtd
    cout << "Height=" << img.height << endl;
    cout << "Width=" << img.width << endl;

    CImgDisplay imgDisplay;
    CImgDisplay imgStatDisplay;
    #endif

    int xx,yy;
    for (int x = 0; x < img.dimx(); x++) {
        for (int y = 0; y < img.dimy(); y++) {
            xx=(img(x,y,0)+img(x,y,1)+img(x,y,2))/3;
            yy=(img.at(x+xL,y+yL,0)+img.at(x+xL,y+yL,1)+img.at(x+xL,y+yL,2))/3;

            if((xx>=0)&&(xx<256)&&(yy>=0)&&(yy<256))
            {
                imgStat.at(xx,yy)++;
            }
            #ifdef _debugSgtd
            else
                cout << xx << " " << yy << " \n";
            #endif
        }
    }
    //Fator de normalização comentado para poder aplicar sgtd em várias imagens antes de normalizar e
    //calcular os descritores
    //imgStat/=(img.dimx()*img.dimy());
    #ifdef _debugSgtd
  //  if (displayImg) {
        imgDisplay.display(img);
        CImg<T> imgTemp=imgStat;
        imgTemp.resize(256,256);
        imgTemp.normalize(0,255);
        imgStatDisplay.display(imgTemp);
        getch();
  //  }
    #endif
}

template<class T> void sgtds(const cimg_library::CImg<T> &img, cimg_library::CImg<float> &imgStat, int L = 1, bool displayImg = false) {
    using namespace std;
    using namespace cimg_library;
//#define _debugSgtds
    #ifdef _debugSgtds
    cout << "Height=" << img.height << endl;
    cout << "Width=" << img.width << endl;

    CImgDisplay imgDisplay;
    CImgDisplay imgStatDisplay;
    #endif
    int N=imgStat.dimx();
    int c0,c1,c2,c3,c4;
    for (int x = 0; x < img.dimx(); x++) {
        for (int y = 1; y < img.dimy(); y++) {
            c0=0;
            c1=0;
            c2=0;
            c3=0;
            c4=0;
            for(int ch=0;ch<img.dimv();ch++)
            {

                c0+=img.at(x,y,ch);
                c1+=img.at(x+L,y,ch);
                c2+=img.at(x,y+L,ch);
                c3+=img.at(x-L,y,ch);
                c4+=img.at(x,y-L,ch);
            }
            c0/=img.dimv();
            c1/=img.dimv();
            c2/=img.dimv();
            c3/=img.dimv();
            c4/=img.dimv();
//            c0=(img(x,y,0)+img(x,y,1)+img(x,y,2))/3;
//            c1=(img(x+L,y,0)+img(x+L,y,1)+img(x+L,y,2))/3;
//            c2=(img(x,y+L,0)+img(x,y+L,1)+img(x,y+L,2))/3;
//            c3=(img(x-L,y,0)+img(x-L,y,1)+img(x-L,y,2))/3;
//            c4=(img(x,y-L,0)+img(x,y-L,1)+img(x,y-L,2))/3;


            if((c0>=0)&&(c0<N)&&(c1>=0)&&(c1<N))
            {
                imgStat.at(c0,c1)++;
                imgStat.at(c0,c2)++;
                imgStat.at(c0,c3)++;
                imgStat.at(c0,c4)++;
            }
            #ifdef _debugSgtds
            else
                cout << c0 << " " << c1 << " \n";
            #endif
        }
    }
    //Fator de normalização comentado para poder aplicar sgtd em várias imagens antes de normalizar e
    //calcular os descritores
    //imgStat/=(img.dimx()*img.dimy());
    #ifdef _debugSgtds
  //  if (displayImg) {
        imgDisplay.display(img);
        CImg<T> imgTemp=imgStat;
        imgTemp.resize(256,256);
        imgTemp.normalize(0,255);
        imgStatDisplay.display(imgTemp);
        getch();
   // }
    #endif
}

template<class T> void calcHaralickDesc(const cimg_library::CImg<T> &imgStat, MatVector<float> &v) {
    v.zeros(4);
    float e = 1;
    float log2 = log(2.0);
    float N=0;
    float mx=0,my=0,vx=0,vy=0;

    for (int x = 0; x < imgStat.dimx(); x++)
        for (int y = 0; y < imgStat.dimy(); y++) {
            mx += imgStat(x, y) * x;
            my += imgStat(x, y) * y;
            N+=imgStat(x,y);
        }

    mx /= N;
    my /= N;

    for (int x = 0; x < imgStat.dimx(); x++)
        for (int y = 0; y < imgStat.dimy(); y++) {
            vx += imgStat(x, y) * (x-mx)*(x-mx);
            vy += imgStat(x, y) * (y-my)*(y-my);

    
            v[0] += imgStat(x, y) * imgStat(x, y); // angular second moment, Homogenidade
            v[1] += (x - y)*(x - y) * imgStat(x, y); //Contrast
            v[2] += imgStat(x,y)*(x-mx)*(y-my);
            v[3] += imgStat(x, y) * log(imgStat(x, y) + e) / log2;
        }
//    vx /= N;
//    vy /= N;
    v[2]/=sqrt(vx*vy);
    //v[7]*=-1.0;
}

template<class T> void grayQuantization(cimg_library::CImg<T> &img, int N) {
    img.normalize(0,N-1);
    float dmin=N+1,d;
    int index=0;

    for(int x=0;x<img.dimx();x++)
        for(int y=0;y<img.dimy();y++)
        {
            dmin=N+1;
            for(float i=0;i<N;i++)
            {
                float l=(img(x,y,0)+img(x,y,1)+img(x,y,2))/3;
                d=abs(l-i);
                if(dmin>d)
                {
                    dmin=d;
                    index=i;
                }
            }
            img(x,y,0)=index;
            img(x,y,1)=index;
            img(x,y,2)=index;
        }
}

template<class T> void grayQuantization(cimg_library::CImg<T> &img, int N,int ch) {
    img.normalize(0,N-1);
    float dmin=N+1,d;
    int index=0;

    for(int x=0;x<img.dimx();x++)
        for(int y=0;y<img.dimy();y++)
        {
            dmin=N+1;
            for(float i=0;i<N;i++)
            {
                float l=img(x,y,ch);
                d=abs(l-i);
                if(dmin>d)
                {
                    dmin=d;
                    index=i;
                }
            }
            img(x,y,ch)=index;
        }
}

template<class T1,class T2> void decompose(const cimg_library::CImg<T1> &img,cimg_library::CImg<T2> &ch0,cimg_library::CImg<T2> &ch1,cimg_library::CImg<T2> &ch2)
{
    ch0.assign(img.dimx(),img.dimy(),1,1);
    ch1.assign(img.dimx(),img.dimy(),1,1);
    ch2.assign(img.dimx(),img.dimy(),1,1);

    for(int x=0;x<img.dimx();x++)
        for(int y=0;y<img.dimy();y++)
        {
            ch0(x,y)=img(x,y,0);
            ch1(x,y)=img(x,y,1);
            ch2(x,y)=img(x,y,2);
        }
};

template<class T1,class T2> void compose(const cimg_library::CImg<T2> &ch0,const cimg_library::CImg<T2> &ch1,const cimg_library::CImg<T2> &ch2,cimg_library::CImg<T1> &img)
{
    img.assign(ch0.dimx(),ch0.dimy(),1,3);
    for(int x=0;x<img.dimx();x++)
        for(int y=0;y<img.dimy();y++)
        {
            img(x,y,0)=ch0(x,y);
            img(x,y,1)=ch1(x,y);
            img(x,y,2)=ch2(x,y);
        }
};

template<class T> void randomSampler(const cimg_library::CImg<T> &img,cimg_library::CImg<T> &sample)
{
    int Lx=img.dimx()-sample.dimx();
    int Ly=img.dimy()-sample.dimy();
    int rLx=MatUtils::unoise(0,Lx);
    int rLy=MatUtils::unoise(0,Ly);

    #ifdef _debugRandomSampler
    std::cout << "x=" << rLx << " y=" << rLy << "\n";
    #endif
    for(int x=0;x<sample.dimx();x++)
        for(int y=0;y<sample.dimy();y++)
        {
            sample(x,y,0)=img(x+rLx,y+rLy,0);
            sample(x,y,1)=img(x+rLx,y+rLy,1);
            sample(x,y,2)=img(x+rLx,y+rLy,2);
        }
};

template<class T> void sampler(const cimg_library::CImg<T> &img, MatVector<float> &v,int Nsamples=10,int dimx=128,int dimy=128,int N=8)
{
    using namespace std;
    using namespace cimg_library;

    float K=dimx*dimy;

    CImg<T> sample(dimx,dimy,3);
    CImg<float> imgStat(N,N,1);
    imgStat.fill(0);
    #ifdef _debugSampler
    MatVector<float> vold;
    CImg<float> imgTemp;
    #endif
    float i;
    for(i=0;i<Nsamples;i++)
    {
        randomSampler(img,sample);
        grayQuantization(sample,N);
        sgtds(sample, imgStat, 1);
        #ifdef _debugSampler
        imgTemp=imgStat;
        imgTemp/=(K*(i+1));
        if(i>0)
            vold=v;
        calcHaralickDesc(imgTemp,v);
        if(i>0)
        {
            vold.sub(v);
            cout << vold.norm() << " m=" << imgStat.mean() << " mm=" << imgTemp.mean() << endl;
        }
        #endif
    }

    imgStat/=(K*i);
    calcHaralickDesc(imgStat,v);
};

template<class T> void composePattern2(const cimg_library::CImg<T> &img1,const cimg_library::CImg<T> &img2,cimg_library::CImg<T> &img3)
{
    int Lx=img1.dimx()+img2.dimx();
    int Ly=MIN(img1.dimy(),img2.dimy());
    int Nch=MIN(img1.dimv(),img2.dimv());
    
    img3.assign(Lx,Ly,1,Nch);

    for (int x = 0; x < img1.dimx(); x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img3(x,y,ch)=img1(x,y,ch);
            }

    for (int x = img1.dimx(); x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img3(x,y,ch)=img2(x,y,ch);
            }

}

template<class T> void composeGroundTruthPattern2(const cimg_library::CImg<T> &img1,const cimg_library::CImg<T> &img2,cimg_library::CImg<T> &img3)
{
    int Lx=img1.dimx()+img2.dimx();
    int Lx2=img1.dimx();
    int Ly=MIN(img1.dimy(),img2.dimy());
    int Nch=MIN(img1.dimv(),img2.dimv());

    img3.assign(Lx,Ly,1,Nch);

    for (int x = 0; x < Lx2; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img3(x,y,ch)=0;
            }

    for (int x = Lx2; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img3(x,y,ch)=1;
            }

}

template<class T> void composePattern4(const cimg_library::CImg<T> &img1,\
const cimg_library::CImg<T> &img2,\
const cimg_library::CImg<T> &img3,\
const cimg_library::CImg<T> &img4,\
cimg_library::CImg<T> &img5)
{
    int Lx=MIN(MIN(img1.dimx(),img2.dimx()),MIN(img3.dimx(),img4.dimx()));
    int Lx2=Lx+Lx;
    int Ly=MIN(MIN(img1.dimy(),img2.dimy()),MIN(img3.dimy(),img4.dimy()));
    int Ly2=Ly+Ly;
    int Nch=MIN(MIN(img1.dimv(),img2.dimv()),MIN(img3.dimv(),img4.dimv()));

    img5.assign(Lx2,Ly2,1,Nch);

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img5(x,y,ch)=img1(x,y,ch);
            }

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img5(x+Lx,y,ch)=img2(x,y,ch);
            }

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img5(x,y+Ly,ch)=img3(x,y,ch);
            }

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img5(x+Lx,y+Ly,ch)=img4(x,y,ch);
            }

}

const uint8 __defaultGrayLevelForComposeGroundTruthPattern4[]={0,1,2,3};
template<class T> void composeGroundTruthPattern4(const cimg_library::CImg<T> &img1,\
const cimg_library::CImg<T> &img2,\
const cimg_library::CImg<T> &img3,\
const cimg_library::CImg<T> &img4,\
cimg_library::CImg<T> &img5,const uint8 c[]=__defaultGrayLevelForComposeGroundTruthPattern4)
{
    int Lx=MIN(MIN(img1.dimx(),img2.dimx()),MIN(img3.dimx(),img4.dimx()));
    int Lx2=Lx+Lx;
    int Ly=MIN(MIN(img1.dimy(),img2.dimy()),MIN(img3.dimy(),img4.dimy()));
    int Ly2=Ly+Ly;
    int Nch=MIN(MIN(img1.dimv(),img2.dimv()),MIN(img3.dimv(),img4.dimv()));

    img5.assign(Lx2,Ly2,1,Nch);

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img5(x,y,ch)=c[0];
            }

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img5(x+Lx,y,ch)=c[1];
            }

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img5(x,y+Ly,ch)=c[2];
            }

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img5(x+Lx,y+Ly,ch)=c[3];
            }

}

template<class T> void composePattern5(
const cimg_library::CImg<T> &img1,\
const cimg_library::CImg<T> &img2,\
const cimg_library::CImg<T> &img3,\
const cimg_library::CImg<T> &img4,\
const cimg_library::CImg<T> &img5,\
cimg_library::CImg<T> &img6)
{
    int Lx=MIN(MIN(img1.dimx(),img2.dimx()),MIN(img3.dimx(),img4.dimx()));
    int Lx2=Lx+Lx;
    int Ly=MIN(MIN(img1.dimy(),img2.dimy()),MIN(img3.dimy(),img4.dimy()));
    int Ly2=Ly+Ly;
    int Nch=MIN(MIN(img1.dimv(),img2.dimv()),MIN(img3.dimv(),img4.dimv()));

    img6.assign(Lx2,Ly2,1,Nch);

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img6(x,y,ch)=img1(x,y,ch);
            }

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img6(x+Lx,y,ch)=img2(x,y,ch);
            }

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img6(x,y+Ly,ch)=img3(x,y,ch);
            }

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img6(x+Lx,y+Ly,ch)=img4(x,y,ch);
            }

        int R=Lx/2;
        for (int y=0;y<R;y++)
        {
            float rt=(float)y/(float)R;
            float X0=(float)R*sqrt(1-rt*rt);
            for(int x=0;x<X0;x++)
                for (int ch = 0; ch < Nch; ch++) {
                img6(Lx+x,Ly+y,ch)=img5(R+x,R+y,ch);
                img6(Lx+x,Ly-y,ch)=img5(R+x,R-y,ch);
                img6(Lx-x,Ly+y,ch)=img5(R-x,R+y,ch);
                img6(Lx-x,Ly-y,ch)=img5(R-x,R-y,ch);
            }
        }
};


const uint8 __defaultGrayLevelForComposeGroundTruthPattern5[]={0,1,2,3,4};
template<class T> void composeGroundTruthPattern5(
const cimg_library::CImg<T> &img1,\
const cimg_library::CImg<T> &img2,\
const cimg_library::CImg<T> &img3,\
const cimg_library::CImg<T> &img4,\
const cimg_library::CImg<T> &img5,\
cimg_library::CImg<T> &img6,const uint8 c[]=__defaultGrayLevelForComposeGroundTruthPattern5)
{
        int Lx=MIN(MIN(img1.dimx(),img2.dimx()),MIN(img3.dimx(),img4.dimx()));
    int Lx2=Lx+Lx;
    int Ly=MIN(MIN(img1.dimy(),img2.dimy()),MIN(img3.dimy(),img4.dimy()));
    int Ly2=Ly+Ly;
    int Nch=MIN(MIN(img1.dimv(),img2.dimv()),MIN(img3.dimv(),img4.dimv()));

    img6.assign(Lx2,Ly2,1,Nch);


    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img6(x,y,ch)=c[0];
            }

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img6(x+Lx,y,ch)=c[1];
            }

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img6(x,y+Ly,ch)=c[2];
            }

    for (int x = 0; x < Lx; x++)
        for (int y = 0; y < Ly; y++)
            for (int ch = 0; ch < Nch; ch++) {
                img6(x+Lx,y+Ly,ch)=c[3];
            }

        int R=Lx/2;
        for (int y=0;y<R;y++)
        {
            float rt=(float)y/(float)R;
            float X0=(float)R*sqrt(1-rt*rt);
            for(int x=0;x<X0;x++)
                for (int ch = 0; ch < Nch; ch++) {
                img6(Lx+x,Ly+y,ch)=c[4];
                img6(Lx+x,Ly-y,ch)=c[4];
                img6(Lx-x,Ly+y,ch)=c[4];
                img6(Lx-x,Ly-y,ch)=c[4];
            }
        }
};


template<class T> int color2NumericLabes(cimg_library::CImg<T> &img, int numberOfch = -1) {
    using namespace std;
    if ((numberOfch != -1)||(img.dimv()==1)) {

        int ch=0;
        if(numberOfch>0)
            ch=numberOfch;

        int Lx=img.dimx(), Ly=img.dimy();
        map<T, uint> colorMap;
        int i = 0;
        int x, y;
        T temp;

        for (x = 0; x < Lx; x++)
            for (y = 0; y < Ly; y++) {
                temp = img(x, y, ch);
                if (colorMap.find(temp) == colorMap.end())
                    colorMap[temp] = i++;
            }

        for (x = 0; x < Lx; x++)
            for (y = 0; y < Ly; y++) {
                temp = img(x, y, ch);
                img(x,y,ch)=colorMap[temp];
            }

        return i;
    }
    else
        errMsg("Esta opção ainda não foi implementada.");

    return 0;
}

template<class T> void compareImgsCS(const cimg_library::CImg<T> &imgIn, cimg_library::CImg<T> &imgGt, float &cs, float k = 0.75) {
    //Varre cada imagem para contar o número de regiões que existem
    using namespace std;
    using namespace cimg_library;

    CImg<T> img(imgIn);

    //Testa se as imagens apresentam as mesmas dimensões, caso contrário retorna cs=0
    if ((img.dimx() == imgGt.dimx()) && (img.dimy() == imgGt.dimy())) {
        int Lx=img.dimx(),Ly=img.dimy();

        typename map<T, int>::iterator itImg;
        map<T, int> imgSegments; // <Cor,área>

        typename map<T, int>::iterator itGt;
        map<T, int> imgGtSegments;// <Cor,área>

        
        for (int x = 0; x < Lx; x++)
            for (int y = 0; y < Ly; y++) {

                /*Monta um mapa com as cores dos segmentos presentes na imagem teste
                 * e conta a área em pixel que cada região ocupa.
                 */
                itImg = imgSegments.find(img(x, y));
                if (itImg == imgSegments.end())
                    imgSegments[img(x, y)] = 1;
                else
                    itImg->second++;

                /*Monta um mapa com as cores dos segmentos presentes na imagem do
                 * ground truth e conta a área em pixel que cada região ocupa.
                 */
                itGt = imgGtSegments.find(imgGt(x, y));
                if (itGt == imgGtSegments.end())
                    imgGtSegments[imgGt(x, y)] = 1;
                else
                    itGt->second++;
            }
      
      /*Faz rotulação crescente. Substitui os rotulos originais nas imagens
       * imgGt (do ground truth) e img(imagem teste) por rótulos sequenciais
       * em ordem crescentes de acordo com os valores armazenados nos mapas
       * imgGtSegments e imgSegments e copia área em pixel de cada região para 
       * o novo mapa.
       */
      map<int, int> segGt;// <rotulo,área>
      map<int, int> segImg;// <rotulo,área>
      int i=0;
      for (itGt = imgGtSegments.begin(); itGt != imgGtSegments.end(); itGt++)
      {
            for (int x = 0; x < Lx; x++)
            for (int y = 0; y < Ly; y++) {
                if(imgGt(x,y)==itGt->first)
                    imgGt(x,y)=i;
            }
            segGt[i]=itGt->second;
            i++;
      }

      i=0;
      for (itImg = imgSegments.begin(); itImg != imgSegments.end(); itImg++)
      {
            for (int x = 0; x < Lx; x++)
            for (int y = 0; y < Ly; y++) {
                if(img(x,y)==itImg->first)
                    img(x,y)=i;
            }
            segImg[i]=itImg->second;
            i++;
      }


#ifdef _debugCompareImgs
        cout << "N(ImgGt)=" << imgGtSegments.size() << endl;
        cout << "N(Img)=" << imgSegments.size() << endl;
#endif

        /*
         * Monta uma matriz m(M,N) onde as linhas são os números de cores da imagem
         * teste e as colunas são as cores do ground truth. Os elementos
         * aij da matriz armazenam as contagens da interseção
         */
        cs=0;
        int N=imgGtSegments.size(),M=imgSegments.size();
        MatMatrix<int> m(M,N);
        m.zeros();
        map<int,int>::iterator itGts;
        map<int,int>::iterator itImgs;
        for (itGts = segGt.begin(); itGts != segGt.end(); itGts++)
            for (itImgs = segImg.begin(); itImgs != segImg.end(); itImgs++) {

                for (int x = 0; x < Lx; x++)
                    for (int y = 0; y < Ly; y++) {
                        if ((img(x, y) == itImgs->first) && (imgGt(x, y) == itGts->first))
                            m[itImgs->first][itGts->first]++;
                    }
           }

        /*
         * Soma as áreas com uma interseção superior a "k" e divide pelo
         * área total em pixels 
         */
        for(int i=0;i<m.rows();i++)
            for(int j=0;j<m.cols();j++)
            {

                if ((m[i][j] >= k*segGt[j]) && (m[i][j] >= k*segImg[i]))
                    cs+=m[i][j];

            }

        int A=Lx*Ly;
        cs/=A;

#ifdef _debugCompareImgs
        cout << "m=" << m << endl;
        cout << "A=" << A << endl;
        cout << "cs=" << cs << endl;
#endif
    } else {
        errMsg("As imagens não tem o mesmo tamanho!!");
        cs = 0;
    }
}


template<class T> void compareImgsCS(const cimg_library::CImg<T> &imgIn, cimg_library::CImg<T> &imgGt, float &cs, MatMatrix<int> &m, float k = 0.75) {
    //Varre cada imagem para contar o número de regiões que existem
    using namespace std;
    using namespace cimg_library;

    CImg<T> img(imgIn);

    //Testa se as imagens apresentam as mesmas dimensões, caso contrário retorna cs=0
    if ((img.dimx() == imgGt.dimx()) && (img.dimy() == imgGt.dimy())) {
        int Lx=img.dimx(),Ly=img.dimy();

        typename map<T, int>::iterator itImg;
        map<T, int> imgSegments; // <Cor,área>

        typename map<T, int>::iterator itGt;
        map<T, int> imgGtSegments;// <Cor,área>

        
        for (int x = 0; x < Lx; x++)
            for (int y = 0; y < Ly; y++) {

                /*Monta um mapa com as cores dos segmentos presentes na imagem teste
                 * e conta a área em pixel que cada região ocupa.
                 */
                itImg = imgSegments.find(img(x, y));
                if (itImg == imgSegments.end())
                    imgSegments[img(x, y)] = 1;
                else
                    itImg->second++;

                /*Monta um mapa com as cores dos segmentos presentes na imagem do
                 * ground truth e conta a área em pixel que cada região ocupa.
                 */
                itGt = imgGtSegments.find(imgGt(x, y));
                if (itGt == imgGtSegments.end())
                    imgGtSegments[imgGt(x, y)] = 1;
                else
                    itGt->second++;
            }
      
      /*Faz rotulação crescente. Substitui os rotulos originais nas imagens
       * imgGt (do ground truth) e img(imagem teste) por rótulos sequenciais
       * em ordem crescentes de acordo com os valores armazenados nos mapas
       * imgGtSegments e imgSegments e copia área em pixel de cada região para 
       * o novo mapa.
       */
      map<int, int> segGt;// <rotulo,área>
      map<int, int> segImg;// <rotulo,área>
      int i=0;
      for (itGt = imgGtSegments.begin(); itGt != imgGtSegments.end(); itGt++)
      {
            for (int x = 0; x < Lx; x++)
            for (int y = 0; y < Ly; y++) {
                if(imgGt(x,y)==itGt->first)
                    imgGt(x,y)=i;
            }
            segGt[i]=itGt->second;
            i++;
      }

      i=0;
      for (itImg = imgSegments.begin(); itImg != imgSegments.end(); itImg++)
      {
            for (int x = 0; x < Lx; x++)
            for (int y = 0; y < Ly; y++) {
                if(img(x,y)==itImg->first)
                    img(x,y)=i;
            }
            segImg[i]=itImg->second;
            i++;
      }


#ifdef _debugCompareImgs
        cout << "N(ImgGt)=" << imgGtSegments.size() << endl;
        cout << "N(Img)=" << imgSegments.size() << endl;
#endif

        /*
         * Monta uma matriz m(M,N) onde as linhas são os números de cores da imagem
         * teste e as colunas são as cores do ground truth. Os elementos
         * aij da matriz armazenam as contagens da interseção
         */
        cs=0;
        int N=imgGtSegments.size(),M=imgSegments.size();
        //MatMatrix<int> m(M,N);
        m.size(M,N);
        m.zeros();
        map<int,int>::iterator itGts;
        map<int,int>::iterator itImgs;
        for (itGts = segGt.begin(); itGts != segGt.end(); itGts++)
            for (itImgs = segImg.begin(); itImgs != segImg.end(); itImgs++) {

                for (int x = 0; x < Lx; x++)
                    for (int y = 0; y < Ly; y++) {
                        if ((img(x, y) == itImgs->first) && (imgGt(x, y) == itGts->first))
                            m[itImgs->first][itGts->first]++;
                    }
           }

        /*
         * Soma as áreas com uma interseção superior a "k" e divide pelo
         * área total em pixels 
         */
        for(int i=0;i<m.rows();i++)
            for(int j=0;j<m.cols();j++)
            {

                if ((m[i][j] >= k*segGt[j]) && (m[i][j] >= k*segImg[i]))
                    cs+=m[i][j];

            }

        int A=Lx*Ly;
        cs/=A;

#ifdef _debugCompareImgs
        cout << "m=" << m << endl;
        cout << "A=" << A << endl;
        cout << "cs=" << cs << endl;
#endif
    } else {
        errMsg("As imagens não tem o mesmo tamanho!!");
        cs = 0;
    }
}

/*template<class T> float compareImgsRand(const cimg_library::CImg<T> &imgIn, cimg_library::CImg<T> &imgGt, MatMatrix<int> &m) {

    using namespace std;
    
    int Nx=imgGt.dimx();
    int Ny=imgGt.dimy();
    
    if((imgIn.dimx()==Nx)&&(imgIn.dimy()==Ny))
    {
        float r=0;
        m.size(2,2);
        
        for(int x=0;x<Nx;x++)
            for(int y=0;y<Nx;y++)
            {
                imgIn(x,y)=
            }
    
        return r;
    }
    
    return 0;
}*/

//Esta função só é válida para imagens em tons de cinza
template<class T> void compareImgs(const cimg_library::CImg<T> &img, cimg_library::CImg<T> &imgGt, float &me, float &nu) {
    //Varre cada imagem para contar o número de regiões que existem
    using namespace std;

    set<T> imgSegments;

    for (int x = 0; x < img.dimx(); x++)
        for (int y = 0; y < img.dimy(); y++) {
            if (imgSegments.find(img(x, y)) == imgSegments.end())
                imgSegments.insert(img(x, y));
        }



    set<T> imgGtSegments;

    for (int x = 0; x < imgGt.dimx(); x++)
        for (int y = 0; y < imgGt.dimy(); y++) {
            if (imgGtSegments.find(imgGt(x, y)) == imgGtSegments.end())
                imgGtSegments.insert(imgGt(x, y));
        }

#ifdef _debugCompareImgs
    cout << "N(ImgGt)=" << imgGtSegments.size() << endl;
    cout << "N(Img)=" << imgSegments.size() << endl;
#endif

    typename set<T>::iterator it;
    typename set<T>::iterator itMin;
    typename set<T>::iterator itImg;
    typename set<T>::iterator itImgMin;

    nu = 0;
    float mme = 0;
    float temp = 0;

    int i = 0;
    for (itImg = imgSegments.begin(); itImg != imgSegments.end(); itImg++) {
        nu = MAX(nu, calcNU<uint8>(img, &(*itImg)));
    }

    i = 0;
    while ((!imgGtSegments.empty()) && (!imgSegments.empty())) {
        me = 2;
        for (it = imgGtSegments.begin(); it != imgGtSegments.end(); it++)
            for (itImg = imgSegments.begin(); itImg != imgSegments.end(); itImg++) {
                temp = calcME(img, *itImg, imgGt, *it);
                if (me > temp) {
                    me = temp;
                    itImgMin = itImg;
                    itMin = it;
                }
            }
        imgGtSegments.erase(itMin);
        imgSegments.erase(itImgMin);
        mme += me;
        i++;
    }

    me = mme / i;
#ifdef _debugCompareImgs
    cout << "me=" << me << endl;
    cout << "nu=" << nu << endl;
#endif
}

template<class T> std::string info(cimg_library::CImg<T> &img,std::string label="dimXYZV=")
{
    using namespace std;
    ostringstream infoText;
    infoText << label;
    infoText << "[ " << img.dimx() <<" " << img.dimy() << " " << img.dimz() \
    << " " << img.dimv() << " ];" <<endl;
    return infoText.str();
};

#endif	/* _CIMGUTILITIES_H */

