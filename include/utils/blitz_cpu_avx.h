#ifndef INCLUDE_UTIL_BLITZ_CPU_AVX_H_
#define INCLUDE_UTIL_BLITZ_CPU_AVX_H_

#include <immintrin.h>

namespace blitz {

template <typename DType>
union BlitzAVXReg;

template<typename DType>
inline void BlitzAVXBroadcast(const DType* addr, BlitzAVXReg<DType>* reg);

template<typename DType>
inline void BlitzAVXLoad(const DType* addr, BlitzAVXReg<DType>* reg);

template<typename DType>
inline void BlitzAVXStore(const DType* addr, BlitzAVXReg<DType>* reg);

template<typename DType>
inline void BlitzAVXMax(BlitzAVXReg<DType>* left,
  BlitzAVXReg<DType>* right, BlitzAVXReg<DType>* output);

template<typename DType>
inline void BlitzAVXMin(BlitzAVXReg<DType>* left,
  BlitzAVXReg<DType>* right, BlitzAVXReg<DType>* output);

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_CPU_AVX_H_
