#pragma once
#include <type_traits>
#include <cmath>
#ifdef CERES_FOUND
  #include <ceres/ceres.h>
#endif

namespace qmri {

// -------- SSFP steady-state signal --------
template <typename T>
inline T ssfp(const T& M0, const T& T1, const T& T2, const T& B1, const T& F0, const T& alpha, const T& tr, const T& phase) {
    
    const T e1 = my_exp(-tr/T(T1));
    const T e2 = my_exp(-tr/T2);
    const T phi = 2.00*PI*tr*T(F0);
    
    const T ca  = my_cos(T(B1)*alpha);
    const T sa  = my_sin(T(B1)*alpha);

    const T beta = phase*deg2rad + phi;
    const T cb  = my_cos(beta);
    const T sb  = my_sin(beta);

    //Non-zero TE corrected signal
    const T denom = (1.00-(e1*ca))*(1.00-(e2*cb)) - (e2*(e1-ca)*(e2-cb));
    const T Mx = M0*(e2*(1.00-e1)*sa*sb)/denom;
    const T My = M0*(e2*(1.00-e1)*(cb-e2)*sa)/denom;
    
    return sqrt(Mx*Mx + My*My);
    
}




} // namespace qmri
