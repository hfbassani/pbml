
#include <list>
#include <MatVector.h>

#include <cimgUtilities.h>

// a predicate implemented as a class:
class compare_colorsVectors
{
public:
    union TColor {
        uint8 colors[3];
        uint32 colorNumber;
    };

  bool operator() (const MatVector<> &v0, const MatVector<> &v1)
  {
      TColor colorV0,colorV1;

      colorV0.colors[0]=v0[0];
      colorV0.colors[1]=v0[1];
      colorV0.colors[2]=v0[2];

      colorV1.colors[0]=v1[0];
      colorV1.colors[1]=v1[1];
      colorV1.colors[2]=v1[2];

      if(colorV0.colorNumber<colorV1.colorNumber)
          return true;

      return false;
  }
};

void img2threshold(cimg_library::CImg<unsigned char> &img, int limiar, int x0, int y0, int x1, int y1) {
    char black[] = {0, 0, 0};
    img.threshold(limiar);
    //img=img*255;
    if (!((x0 == 0) && (x1 == 0) && (y0 == 0) && (y1 == 0)))
        img.draw_rectangle(x0, y0, x1, y1, black);
}

void img2outline(cimg_library::CImg<unsigned char> &img, const int maskSize) {
    using namespace std;
    using namespace cimg_library;

    CImg<unsigned char> imgTemp, mask(maskSize, maskSize);
    mask.fill(255);
    imgTemp = img;
    img.erode(mask);
    img = img - imgTemp;
}



void img2colorList(const cimg_library::CImg<unsigned char> &img, MatVectorList<float> &vList, bool randomSamples, int samplesCount, const int index[]) {
    MatVector<float> v(3);

    if(randomSamples)
    {
        int counter=0;
        MatVector<int> randomx(img.width);
        MatVector<int> randomy(img.height);
        randomx.range(0,img.width-1);
        randomy.range(0,img.height-1);
        randomx.shuffler();
        randomy.shuffler();
        for (int x = 0; x < img.width; x++)
            for (int y = 0; y < img.height; y++) {
                v[index[0]] = img(randomx[x],randomy[y], 0);
                v[index[1]] = img(randomx[x],randomy[y], 1);
                v[index[2]] = img(randomx[x],randomy[y], 2);
                vList.push_back(v);

                if(counter>=samplesCount)
                {
                    x=img.width;
                    break;
                }

                counter++;
            }
    }
    else
    {
        for (int x = 0; x < img.width; x++)
            for (int y = 0; y < img.height; y++) {
                v[index[0]] = img(x, y, 0);
                v[index[1]] = img(x, y, 1);
                v[index[2]] = img(x, y, 2);
                vList.push_back(v);
            }

    }

    vList.sort(compare_colorsVectors());
    vList.unique();
}

void getImgStatistics(const cimg_library::CImg<unsigned char> &img, MatVector<float> &vstat)
{

    /*
     * vstat = [ ]
     * v[0] = média(canal0)
     * v[1] = var(canal0)
     *
     * v[2] = média(canal1)
     * v[3] = var(canel1)
     *
     * v[4] = média(canal2)
     * v[5] = var(canal2)
     */
    double rm=0,gm=0,bm=0;
    double rv=0,gv=0,bv=0;
    int N=img.width*img.height;
    for(int x=0;x<img.width;x++)
    {
        for(int y=0;y<img.height;y++)
        {
            rm+=img(x,y,0);
            gm+=img(x,y,1);
            bm+=img(x,y,2);
        }
    }
    rm/=N;
    gm/=N;
    bm/=N;

    for(int x=0;x<img.width;x++)
    {
        for(int y=0;y<img.height;y++)
        {
            rv+=(img(x,y,0)-rm)*(img(x,y,0)-rm);
            gv+=(img(x,y,1)-gm)*(img(x,y,1)-gm);
            bv+=(img(x,y,2)-bm)*(img(x,y,2)-bm);
        }
    }
    rv/=N;
    gv/=N;
    bv/=N;

    vstat.size(6);
    vstat[0]=rm;
    vstat[1]=sqrt(rv);

    vstat[2]=gm;
    vstat[3]=sqrt(gv);

    vstat[4]=bm;
    vstat[5]=sqrt(bv);
}

void getImgHistogram(const cimg_library::CImg<unsigned char> &img, MatVector<float> &v, bool imgMonocrome) {

    int width = img.width;
    int height = img.height;
    float a=width*height;

    if (imgMonocrome) {
        v.zeros(256);
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                v[img(x, y, 0)]++;
            }

    } else {
        v.zeros(256 * 3);
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                v[img(x, y, 0)]++;
                v[img(x, y, 1) + 256]++;
                v[img(x, y, 2) + 512]++;
            }
    }
    v.div((float)a);
}

void getImgHistogram(const cimg_library::CImg<unsigned char> &img, MatVector<float> &v,int N, bool imgMonocrome) {

    int width = img.width;
    int height = img.height;
    float a=width*height;
    int NN=ceil(256.0/N);
    if (imgMonocrome) {
        v.zeros(N);
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                int i=floor(img(x, y, 0)/NN);
                v[i]++;
            }

    } else {
        v.zeros(N * 3);
        int N2=N*2;
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                v[floor(img(x, y, 0)/NN)]++;
                v[floor(img(x, y, 1)/NN) + N]++;
                v[floor(img(x, y, 2)/NN) + N2]++;
            }
    }
    v.div((float)a);
}

void getImgMoms(const cimg_library::CImg<unsigned char> &img, MatVector<float> &vmom,int color,bool mono) {
    uint R = img.width;
    uint C = img.height;
    double mtemp;

    double xc = 0;
    double yc = 0;
    double s = 0;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            
            if(mono)
            {
                mtemp+=img(i,j,0);
                mtemp+=img(i,j,1);
                mtemp+=img(i,j,2);
                mtemp/=3;
            }
            else
                mtemp=img(i,j,color);

            xc += mtemp * j;
            yc += mtemp * i;
            s += mtemp;
        }
    }
    xc /= s;
    yc /= s;

    double mom00=0;
    double mom11=0;
    double mom20=0;
    double mom02=0;
    double mom22=0;
    double mom12=0;
    double mom21=0;
    double mom30=0;
    double mom03=0;
    double mom31=0;
    double mom13=0;
    double mom40=0;
    double mom04=0;
    
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {

            if(mono)
            {
                mtemp+=img(i,j,0);
                mtemp+=img(i,j,1);
                mtemp+=img(i,j,2);
                mtemp/=3;
            }
            else
                mtemp=img(i,j,color);
            
            mom00 += mtemp;
            mom11 += (mtemp * j-xc)*(mtemp * i-yc)*mtemp;
            mom20 += pow(mtemp * j-xc,2)*mtemp;
            mom02 += pow(mtemp * i-yc,2)*mtemp;
            mom22 += pow(mtemp * j-xc,2)*pow(mtemp * i-yc,2)*mtemp;
            mom12 += (mtemp * j-xc)*pow(mtemp * i-yc,2)*mtemp;
            mom21 += pow(mtemp * j-xc,2)*(mtemp * i-yc)*mtemp;
            mom30 += pow(mtemp * j-xc,3)*mtemp;
            mom03 += pow(mtemp * i-yc,3)*mtemp;
            mom31 += pow(mtemp * j-xc,3)*(mtemp * i-yc)*mtemp;
            mom13 += (mtemp * j-xc)*pow(mtemp * i-yc,3)*mtemp;
            mom40 += pow(mtemp * j-xc,4)*mtemp;
            mom04 += pow(mtemp * i-yc,4)*mtemp;
        }
    }
    vmom.size(6);
    vmom[0]=(1/pow(mom00,4))*(mom20*mom02-pow(mom11,2));

    vmom[1]=(1/pow(mom00,10))*(pow(mom30,2)*pow(mom03,2)-6*mom30*mom21*mom12*mom03+4*mom30*pow(mom12,3)
            +4*mom03*pow(mom21,3)-3*pow(mom21,2)*pow(mom12,2));

    vmom[2]=(1/pow(mom00,7))*(mom20*(mom21*mom03-pow(mom12,2))-mom11*(mom30*mom03-mom21*mom12)
            +mom02*(mom30*mom12-pow(mom21,2)));

    vmom[3]=(1/pow(mom00,11))*(pow(mom20,3)*pow(mom03,2)-6*pow(mom20,2)*mom11*mom12*mom03-6*pow(mom20,2)*mom02*mom21*mom03
                +9*pow(mom20,2)*mom02*pow(mom12,2)+12*mom20*pow(mom11,2)*mom21*mom03
                +6*mom20*mom11*mom02*mom30*mom03-18*mom20*mom11*mom02*mom21*mom12
                -8*pow(mom11,3)*mom30*mom03-6*mom20*pow(mom02,2)*mom30*mom12+9*mom20*pow(mom02,2)*pow(mom21,2)
                +12*pow(mom11,2)*mom02*mom30*mom12-6*mom11*pow(mom02,2)*mom30*mom21+pow(mom02,3)*pow(mom30,2));

    vmom[4]=(1/pow(mom00,6))*(mom40*mom04-4*mom31*mom13+3*pow(mom22,2));

    vmom[5]=(1/pow(mom00,9))*(mom40*mom04*mom22+2*mom31*mom22*mom13-mom40*pow(mom13,2)-mom04*pow(mom31,2)-pow(mom22,3));
    vmom.vabs().vlog().replaceInf(1000);

};



void img2pointList(const cimg_library::CImg<unsigned char> &img, TVectorList &vList, int limiar, float z, float ax, float ay, float bx, float by) {

    for (int i = 0; i < img.dimy(); i++)
        for (int j = 0; j < img.dimx(); j++) {
            int m = (img(i, j, 0) + img(i, j, 1) + img(i, j, 2)) / 3;
            if (m > limiar) {
                MatVector<float> v(3);
                v[0] = ax * j + bx;
                v[1] = ay * i + by;
                v[2] = z;
                vList.push_back(v);
            }

        }
}

void img2pointList(const cimg_library::CImg<unsigned char> &img, TVectorList &vList, int limiar, float ax, float ay, float bx, float by) {

    for (int i = 0; i < img.dimy(); i++)
        for (int j = 0; j < img.dimx(); j++) {
            int m = (img(i, j, 0) + img(i, j, 1) + img(i, j, 2)) / 3;
            if (m > limiar) {
                MatVector<float> v(2);
                v[0] = ax * j + bx;
                v[1] = ay * i + by;
                vList.push_back(v);
            }

        }
}

void scannerImg(const cimg_library::CImg<unsigned char> &img, int xL0, int yL0, int xL1, int yL1, void f(int x, int y, const cimg_library::CImg<unsigned char> &imgWindow), int xLstep, int yLstep, bool borderCondition) {
    int width = img.width;
    int height = img.height;
    cimg_library::CImg<unsigned char> imgTemp;

    for (int x = 0; x < width; x += xLstep)
        for (int y = 0; y < height; y += yLstep) {
            imgTemp = img.get_crop(x - xL0, y - yL0, x + xL1, y + yL1, borderCondition);
            f(x, y, imgTemp);
        }
}

void pointList2wrl(std::ofstream &outputFile, TVectorList &vlist) {

    using namespace std;

    outputFile << "#VRML V2.0 utf8\n\
	#tempo de execucao: 0\n\
	Background{ skyColor 1 1 1 }\n\
	Shape{\n\
	appearance Appearance{\n\
	material Material{\n\
	emissiveColor 0 0 0\n\
	diffuseColor 0 0 0\n\
	}\n\
	}\n\
	geometry PointSet {\n\
	coord Coordinate{point [\n";

    TVectorList::iterator it;

    for (it = vlist.begin(); it != vlist.end(); it++)
        outputFile << (*it)[0] << " " << (*it)[1] << " " << (*it)[2] << endl;

    outputFile << "]\n}\n}}" << endl;


}

void img2wrl(std::ofstream &outputfile, cimg_library::CImg<unsigned char> &img, int limiar) {
    using namespace std;
    using namespace cimg_library;

    outputfile << "#VRML V2.0 utf8\n\
	#tempo de execucao: 0\n\
	Background{ skyColor 1 1 1 }\n\
	Shape{\n\
	appearance Appearance{\n\
	material Material{\n\
	emissiveColor 0 0 0\n\
	diffuseColor 0 0 0\n\
	}\n\
	}\n\
	geometry PointSet {\n\
	coord Coordinate{point [\n";

    for (int i = 0; i < img.dimy(); i++)
        for (int j = 0; j < img.dimx(); j++) {
            int m = (img(i, j, 0) + img(i, j, 1) + img(i, j, 2)) / 3;
            if (m > limiar)
                outputfile << j << " " << i << " " << 0 << endl;
        }
    outputfile << "]\n}\n}}" << endl;
}

void imgList2wrl(std::ofstream &outputfile, TNameVector names, int limiar, float scale) {
    using namespace std;
    using namespace cimg_library;

    outputfile << "#VRML V2.0 utf8\n\
	#tempo de execucao: 0\n\
	Background{ skyColor 1 1 1 }\n\
	Shape{\n\
	appearance Appearance{\n\
	material Material{\n\
	emissiveColor 0 0 0\n\
	diffuseColor 0 0 0\n\
	}\n\
	}\n\
	geometry PointSet {\n\
	coord Coordinate{point [\n";

    for (uint k = 0; k < names.size(); k++) {
        CImg<char> img;
        img.load(names[k].c_str());

        for (int i = 0; i < img.dimy(); i++)
            for (int j = 0; j < img.dimx(); j++) {
                int m = (img(i, j, 0) + img(i, j, 1) + img(i, j, 2)) / 3;
                if (m > limiar)
                    outputfile << j << " " << i << " " << k * scale << endl;
            }
    }

    outputfile << "]\n}\n}}" << endl;

}

template<class T1, class T2> void img2Matrix(const cimg_library::CImg<T1> &img, MatMatrix<T2> &m, int channel) {
    m.size(img.height, img.width);
    for (int x = 0; x < img.width; x++)
        for (int y = 0; y < img.height; y++) {
            m[y][x] = img(x, y, channel);
        }
}

unsigned char** makeColorPallet(const int &colors) {

    unsigned char** pallet;
    std::div_t tdiv1, tdiv2;
    tdiv1 = div(colors, 2);
    tdiv2 = div(255, tdiv1.quot);

    pallet = new unsigned char*[colors];
    pallet[0] = new unsigned char[colors * 3];

    for (int i = 1; i < colors; i++)
        pallet[i] = pallet[i - 1] + 3;

    int r, g, b;
    r = 255;
    g = b = 0;
    int N = 0;
    for (int i = 0; i < tdiv1.quot; i++) {
        //cout << "color={ " << r << " ," << g << " ," << b << " }" << endl;
        pallet[N][0]=r;
        pallet[N][1]=g;
        pallet[N][2]=b;

        r -= tdiv2.quot;
        g += tdiv2.quot;
        N++;
    }
    r = 0;
    for (int i = 0; i < tdiv1.quot; i++) {
        //cout << "color={ " << r << " ," << g << " ," << b << " }" << endl;
        pallet[N][0]=r;
        pallet[N][1]=g;
        pallet[N][2]=b;

        g -= tdiv2.quot;
        b += tdiv2.quot;
        N++;
    }
    if (tdiv1.rem > 0) {
        //cout << "color={ " << r << " ," << g << " ," << b << " }" << endl;
        pallet[N][0]=r;
        pallet[N][1]=g;
        pallet[N][2]=b;
    }
//    cout << "N=" << N << endl;

    return pallet;
}

unsigned char** makeMonoPallet(const int colors, const int ch) {

    using namespace std;
    unsigned char** pallet;
    std::div_t tdiv1;
    tdiv1 = div(256,colors);

    pallet = new unsigned char*[colors];
    pallet[0] = new unsigned char[colors * 3];

    for (int i = 1; i < colors; i++)
        pallet[i] = pallet[i - 1] + 3;

    for(int i=0;i<(colors*3);i++)
        pallet[0][i]=100;

    int temp;
    for (int i = 1; i <= colors; i++) {
        temp=tdiv1.quot*i-1;
        //cout << "pallet=" << temp << endl;
        pallet[i-1][ch]=temp;
    }

    return pallet;
}

void destroyPallet(unsigned char** pallet)
{
    delete[] pallet[0];
    delete[] (pallet);
    pallet=NULL;
}


