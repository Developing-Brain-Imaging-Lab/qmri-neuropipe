#pragma once
#include <type_traits>
#include <cmath>
#ifdef CERES_FOUND
  #include <ceres/ceres.h>
#endif

namespace qmri {

// Small helpers to pick sin/cos/exp for either doubles or Ceres Jets
template <typename T>
inline T my_sin(const T& x) {
#ifdef CERES_FOUND
  using ceres::sin;
  return sin(x);
#else
  using std::sin;
  return sin(x);
#endif
}
template <typename T>
inline T my_cos(const T& x) {
#ifdef CERES_FOUND
  using ceres::cos;
  return cos(x);
#else
  using std::cos;
  return cos(x);
#endif
}
template <typename T>
inline T my_exp(const T& x) {
#ifdef CERES_FOUND
  using ceres::exp;
  return exp(x);
#else
  using std::exp;
  return exp(x);
#endif
}
template <typename T>
inline T my_log(const T& x) {
#ifdef CERES_FOUND
  using ceres::log;
  return log(x);
#else
  using std::log;
  return log(x);
#endif
}

// -------- SPGR steady-state signal --------
template <typename T>
inline T spgr(const T& M0, const T& T1, const T& B1, const T& alpha, const T& tr) {
    const T sa = my_sin(B1*alpha);
    const T ca = my_cos(B1*alpha);
    const T E1 = my_exp(-tr / T1);
    
    return M0 * sa * (T(1.) - E1)/(T(1.) - ca * E1);

}

// Optional T2 decay factor
template <typename T>
inline T t2_decay(const T& T2, const T& te) {
  return my_exp(-te / T2);
}

// Convenience SPGR with T2 (just multiplies the base model)
template <typename T>
inline T spgr_with_t2(const T& M0, const T& T1, const T& T2, const T& B1,
                      const T& alpha, const T& tr, const T& te) {
  return spgr(M0, T1, B1, alpha, tr) * t2_decay(T2, te);
}




} // namespace qmri
