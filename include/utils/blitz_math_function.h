#ifndef INCLUDE_UTIL_BLITZ_MATH_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_MATH_FUNCTION_H_

#include <cmath>

namespace blitz {

unsigned int blitz_len_d2b(unsigned int n) {
  unsigned int i, j = 0;
  i = n;
  while (i) {
    i /= 2;
    j++;
  }
  return j;
}

void blitz_magic32(unsigned int nmax, unsigned int d, unsigned int& m, unsigned int& p) {
  unsigned int nc = ((nmax + 1) / d) * d - 1;
  unsigned int nbits = blitz_len_d2b(nmax);
  unsigned int len_bits = 2 * nbits + 1;
  for(p = 0; p < len_bits; ++p) {   
    if(pow(2, p) > nc * (d - 1 -
      static_cast<unsigned int>((pow(2, p) - 1)) % d)) {
      m = (pow(2, p) + d - 1 -
      static_cast<unsigned int>((pow(2, p) - 1)) % d) / d;
      return;
    }   
  }   
}


}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_MATH_FUNCTION_H_
