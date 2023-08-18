///
/// Author: Olivier Stasse
/// Copyright: LAAS-CNRS
/// Date: 2022
///
///
/// Author2: David Bellis
/// Copyright2: Nimble One
/// Date2: Aug 2023
///
#pragma once

#include <Eigen/Eigen>

#if !EIGEN_HAS_CXX11
#define XSTR(x) STR(x)
#define STR(x) #x
#pragma message "The value of EIGEN_MAX_CPP_VER is " XSTR(EIGEN_MAX_CPP_VER)
#pragma message "The value of EIGEN_COMP_CXXVER is " XSTR(EIGEN_COMP_CXXVER)
#error EIGEN_HAS_CXX11 is required.
#endif

#include <unsupported/Eigen/CXX11/Tensor>

#include <mpc-interface/dynamics.hh>

namespace nimbleone {
namespace mpc {

void extend_matrices(Eigen::Tensor<double, 3> &S, Eigen::Tensor<double, 4> &U,
                     unsigned int N, Eigen::MatrixXd *A, Eigen::MatrixXd *B);

} // namespace mpc

} // namespace nimbleone
