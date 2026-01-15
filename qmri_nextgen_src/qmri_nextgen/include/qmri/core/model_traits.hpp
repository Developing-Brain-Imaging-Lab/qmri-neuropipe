#pragma once

namespace qmri {

#define PI 3.1415926535897932384626433
#define deg2rad 0.0174532925199433

#define TI_SCALE 0.95
#define IRSPGR_PD_SCALE 0.975

#define T1_MIN 0.001
#define T1_MAX 10.00
#define T2_MIN 0.001
#define T2_MAX 5.000

#define SPGRWEIGHT 2.75
#define SSFPWEIGHT 1.00
#define SSFPWEIGHT_HIGH 2.15
#define SSFPWEIGHT_LOW 0.55


template<typename M> struct ModelTraits {
  static constexpr bool HasAnalyticJacobian = false;
  static constexpr int  MaxParams = -1;
};


} // namespace qmri
