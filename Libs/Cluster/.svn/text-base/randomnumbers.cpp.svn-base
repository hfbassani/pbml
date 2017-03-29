/* 
 * File:   randomnumbers.cpp
 * Author: hans
 *
 * Created on 23 de Setembro de 2011, 11:14
 */

#include <cstdlib>
#include <math.h>

double randomUniform(double low, double high)
{
    double temp;

    /* swap low & high around if the user makes no sense */
    if (low > high)
    {
    temp = low;
    low = high;
    high = temp;
    }

    /* calculate the random number & return it */
    temp = (rand() / (double(RAND_MAX) + 1.0)) * (high - low) + low;
    return temp;
}

double randomUniform01(){
        return double (rand())/double(RAND_MAX);  /* ranf() is uniform in 0..1 */
}

/*  the method boxMuller implements the Polar form of the Box-Muller Transformation

(c) Copyright 1994, Everett F. Carter Jr.
 Permission is granted by the author to use
 this function for any application provided this
 copyright notice is preserved.
 source: http://www.taygeta.com/random/gaussian.html
*/
double randomGaussian(double m, double s)/* normal random variate generator */
{                                       /* mean m, standard deviation s */
 double x1, x2, w, y1;
 static double y2;
 static int use_last = 0;
 if (use_last){                       /* use value from previous call */
    y1 = y2;
    use_last = 0;
 }
 else {
    do {
       x1 = 2.0 * randomUniform01() - 1.0;
       x2 = 2.0 * randomUniform01() - 1.0;
       w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 );
    w = sqrt( (-2.0 * log( w ) ) / w );
    y1 = x1 * w;
    y2 = x2 * w;
    use_last = 1;
 }
 return( m + y1 * s );
}

