#include "utils/blitz_math_function.h"
#include <cmath>

namespace blitz {

namespace utils {

size_t BlitzLenD2b(size_t n) {
  size_t i, j = 0;
  i = n;
  while (i) {
    i /= 2;
    j++;
  }
  return j;
}

void BlitzMagic32(size_t nmax, size_t d, size_t& m, size_t& p) {
  size_t nc = ((nmax + 1) / d) * d - 1;
  size_t nbits = BlitzLenD2b(nmax);
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

}  // namespace utils

}  // namespace blitz
