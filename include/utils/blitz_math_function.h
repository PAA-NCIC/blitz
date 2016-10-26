#ifndef INCLUDE_UTIL_BLITZ_MATH_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_MATH_FUNCTION_H_

#include <cmath>

namespace blitz {

size_t blitz_len_d2b(size_t n) {
  size_t i, j = 0;
  i = n;
  while (i) {
    i /= 2;
    j++;
  }
  return j;
}

void blitz_magic32(size_t nmax, size_t d, size_t&  m, size_t& p) {
  size_t nc = ((nmax + 1) / d) * d - 1;
  size_t nbits = blitz_len_d2b(nmax);
  size_t len_bits = 2 * nbits + 1;
  for(p = 0; p < len_bits; ++p) {   
    if(pow(2, p) > nc * (d - 1 -
      static_cast<size_t>((pow(2, p) - 1)) % d)) {
      m = (pow(2, p) + d - 1 -
      static_cast<size_t>((pow(2, p) - 1)) % d) / d;
      return;
    }   
  }   
}


}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_MATH_FUNCTION_H_
