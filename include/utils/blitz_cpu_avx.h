#ifndef INCLUDE_UTIL_BLITZ_CPU_AVX_H_
#define INCLUDE_UTIL_BLITZ_CPU_AVX_H_

namespace blitz {

namespace utils {

template <typename DType>
union BlitzAVXReg;

template<typename DType>
inline void BlitzAVXBroadcast(const DType* addr, BlitzAVXReg<DType>* reg);

template<typename DType>
inline void BlitzAVXLoad(const DType* addr, BlitzAVXReg<DType>* reg);

template<typename DType>
inline void BlitzAVXStore(const DType* addr, BlitzAVXReg<DType>* reg);

template<typename DType>
inline void BlitzAVXMax(const BlitzAVXReg<DType>* left, const BlitzAVXReg<DType>* right, BlitzAVXReg<DType>* output);

template<typename DType>
inline void BlitzAVXMin(const BlitzAVXReg<DType>* left, const BlitzAVXReg<DType>* right, BlitzAVXReg<DType>* output);

}

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_CPU_AVX_H_
