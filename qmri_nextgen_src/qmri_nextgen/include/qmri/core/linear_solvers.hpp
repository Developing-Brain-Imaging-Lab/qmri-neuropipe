#pragma once
#include <Eigen/Dense>
#include <cmath>
#include "optimizer_tags.hpp"

namespace qmri {

inline Eigen::Vector2d linear_regression(const Eigen::Ref<const Eigen::VectorXd>& X,
                                         const Eigen::Ref<const Eigen::VectorXd>& Y){
    
    Eigen::Vector2d x;
    Eigen::ArrayXd Xa = X.template cast<double>();
    Eigen::ArrayXd Ya = Y.template cast<double>();
    
    double sumX = Xa.sum();
    double sumY = Ya.sum();
    double sumXX = (Xa*Xa).sum();
    double sumXY = (Xa*Ya).sum();
    
    double a = (Ya.rows()*sumXY)-(sumX*sumY);
    double d = (Ya.rows()*sumXX)-(sumX*sumX);
    
    if (d != 0.00){
        x[0] = a/d;
        x[1] = (sumY-x[0]*sumX)/Y.rows();
    }
    else{
        x[0] = 0.00;
        x[1] = 0.00;
    }
    
    return x;
    
}

inline Eigen::Vector2d ordinary_least_squares(const Eigen::Ref<const Eigen::VectorXd>& X,
                                              const Eigen::Ref<const Eigen::VectorXd>& Y){
    
    Eigen::Matrix<double, Eigen::Dynamic, 2> Xd(X.size(),2);
    Xd.col(0) = X;
    Xd.col(1).setOnes();
    
    return (Xd.transpose()*Xd).partialPivLu().solve(Xd.transpose()*Y);
    
}

inline Eigen::Vector2d weighted_least_squares(const Eigen::Ref<const Eigen::VectorXd>& X,
                                              const Eigen::Ref<const Eigen::VectorXd>& Y,
                                              const Eigen::Ref<const Eigen::VectorXd>& W){
    
    Eigen::Matrix<double, Eigen::Dynamic, 2> Xd(X.size(),2);
    Eigen::MatrixXd Wd = W.asDiagonal();
    
    Xd.col(0) = X;
    Xd.col(1).setOnes();
    
    return (Xd.transpose()*Wd*Xd).partialPivLu().solve(Xd.transpose()*Wd*Y);
    
    
}



} // namespace qmri
