#ifndef LHS_H_
#define LHS_H_

#include <vector>
#include "MatMatrix.h"
#include "MatVector.h"

class LHS {

public:
    
    static double nextDouble() {
        return (double)rand() / RAND_MAX;
    }
    
    static void getPermutation(double min, double max, int N, MatVector<double> &perm) {
            double step = (max - min)/N;

            perm.size(N);
            int i, k;
            // initialization
            for (i = 0; i < N; i++)
                    perm[i] = min + i*step + nextDouble()*step;
            // permutation
            for (i = 0; i < N-1; i++)
            {
                     k = i + (int)(nextDouble()*(N-i));
                     double temp = perm[i];
                     perm[i] = perm[k];
                     perm[k] = temp;
            }
    }

    static void getLHS(const MatMatrix<double> &ranges, MatMatrix<double> &lhs, int N) {
        lhs.size(N, ranges.rows());

        int c=0;
        for (int i=0; i<ranges.rows(); i++) {
            MatVector<double> perm; 
            getPermutation(ranges[i][0], ranges[i][1], N, perm);
                
            for (int r = 0; r < N; r++) {
                    lhs[r][c] = perm[r];
            }
            c++;
        }
    }
};
#endif /*LHS_H_*/
