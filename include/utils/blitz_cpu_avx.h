#ifndef INCLUDE_UTIL_BLITZ_CPU_AVX_H_
#define INCLUDE_UTIL_BLITZ_CPU_AVX_H_

namespace blitz {

namespace utils {

template <typename DType>
union AVXReg;

template<typename DType>
inline void AVXBroadcast(const DType* addr, AVXReg<DType>* reg);

template<typename DType>
inline void AVXLoad(const DType* addr, AVXReg<DType>* reg);

template<typename DType>
inline void AVXStore(const DType* addr, AVXReg<DType>* reg);

template<typename DType>
inline void AVXMax(const AVXReg<DType>* left, const AVXReg<DType>* right, AVXReg<DType>* output);

template<typename DType>
inline void AVXMin(const AVXReg<DType>* left, const AVXReg<DType>* right, AVXReg<DType>* output);

}

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_CPU_AVX_H_
