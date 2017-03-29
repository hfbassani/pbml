/* 
 * File:   dsWeightsDims.cpp
 * Author: hans
 * 
 * Created on 26 de Setembro de 2011, 17:51
 */

#include "dsWeightsDims.h"


bool sortDSWeightsDims(const DSWeightsDims& dswd1, const DSWeightsDims& dswd2) {
   return dswd1.dsWeight > dswd2.dsWeight;
}

bool sortDSWeightsIndex(const DSWeightsDims& dswd1, const DSWeightsDims& dswd2) {
   if (dswd1.dimIndex == dswd2.dimIndex)
        return dswd1.dsWeight > dswd2.dsWeight;

   return dswd1.dimIndex > dswd2.dimIndex;
}

bool operator==(const DSWeightsDims& dswd1, const DSWeightsDims& dswd2) {
    return (dswd1.dimIndex==dswd2.dimIndex);
}
