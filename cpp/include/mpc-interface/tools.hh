///
/// Gecko - Tools
/// Author: Olivier Stasse
/// Copyright: LAAS-CNRS
/// Date: 2022
///
///
/// Author2: David Bellis
/// Copyright2: Nimble One
/// Date2: Aug 2023
///

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

namespace gecko {
namespace tools {

void extend_matrices(Eigen::Tensor<double, 3> &S, Eigen::Tensor<double, 4> &U,
                     unsigned int N, Eigen::MatrixXd *A, Eigen::MatrixXd *B);


void update_step_matrices(
  void ** objects,
  std::map<std::string, double> & kargs);

inline void do_not_update(
  void ** objects,
  std::map<std::string, double> & kargs)
{
    (void)objects;
    (void)kargs;
}

} // namespace tools

} // namespace gecko
