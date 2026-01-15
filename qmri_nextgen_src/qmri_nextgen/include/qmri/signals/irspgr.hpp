#pragma once
#include <type_traits>
#include <cmath>
#ifdef CERES_FOUND
  #include <ceres/ceres.h>
#endif

namespace qmri {

// -------- IR‑SPGR single‑readout approximation --------
template <typename T>
inline T irspgr(const T& M0, const T& T1, const T& B1, const T& alpha, const T& tr, const T& ti, const T& etl) {
    
    const T sa = my_sin(B1*alpha), ca = my_cos(B1*alpha);
    const T eta = -T(0.9); // Inversion efficiency defined as -1 < eta < 0

    const T      R1s = ((1./T1) - my_log(ca) / tr);
    const T      M0s = M0 * (1.0 - my_exp(-tr/T1)) / (1.0 - my_exp(-tr*R1s));
    const T      A_1 = M0s * (1.0 - my_exp(-(etl*tr)*R1s));
    const T      A_3 = M0 * (1.0 - my_exp(-ti/T1));

    const T      B_1 = my_exp(-(etl*tr)*R1s);
    const T      B_3 = eta * my_exp(-ti/T1);

    const T A  = A_3 + A_1 * B_3;
    const T B  = B_1 * B_3;
    const T M1 = A / (1.0 - B);

    return (M0s + (M1 - M0s))*sa;

}

} // namespace qmri
