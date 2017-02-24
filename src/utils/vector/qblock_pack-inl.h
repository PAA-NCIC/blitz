#undef SRC_UTILS_VECTOR_QBLOCK_PACK_INL_H_
#ifndef SRC_UTILS_VECTOR_QBLOCK_PACK_INL_H_
#define SRC_UTILS_VECTOR_QBLOCK_PACK_INL_H_

size_t aq = iq;
size_t ap = ip;
for (size_t bpq = 0; bpq < PQBLOCK; ++bpq) {
  int ih_index = static_cast<int>(ap * str_h) - static_cast<int>(pad_h) + static_cast<int>(r);
  int iw_index = static_cast<int>(aq * str_w) - static_cast<int>(pad_w) + static_cast<int>(s);
  if (ih_index >= 0 && ih_index < static_cast<int>(H)) {
    if (iw_index >= 0 && iw_index < static_cast<int>(W)) {
      for (size_t bc = 0; bc < rc; ++bc) {
        I_pack[bpq * CBLOCK + bc] = ACCESS_INPUT_NHWC(n, ih_index, iw_index, (ic + bc));
      }
    } else {
      for (size_t bc = 0; bc < CBLOCK; ++bc) {
        I_pack[bpq * CBLOCK + bc] = 0;
      }
    }
  } else {
    for (size_t bc = 0; bc < CBLOCK; ++bc) {
      I_pack[bpq * CBLOCK + bc] = 0;
    }
  }
  aq += 1;
  if (aq >= Q) {
    ap += 1;
    aq = 0;
  } 
}

#endif  // SRC_UTILS_VECTOR_QBLOCK_PACK_INL_H_
