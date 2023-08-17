#include <iostream>
#include <cassert>

// Boost Headers
#include <boost/test/unit_test.hpp>

// Project headers
#include <mpc-interface/tools.hh>

// the Agent helps us run tests
#include "testAgent.h"

using namespace Eigen;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(test_extend_matrices_body) {
  int a0 = 3;
  int a1 = 3;
  Eigen::MatrixXd A(a0, a1);
  A.row(0) << 1., 0.1, 0.005;
  A.row(1) << 0., 1., 0.1;
  A.row(2) << 0., 0., 1.;

  int b0 = 3;
  int b1 = 1;
  Eigen::MatrixXd B(b0, b1);
  B.row(0) << 0.00016667;
  B.row(1) << 0.005;
  B.row(2) << 0.1;

  std::stringstream ssA;
  write_output<Eigen::Matrix<double, Dynamic, Dynamic>>(
    "output/test_extend_matrices_body_A.oc", get_output(ssA, A));
  std::stringstream ssB;
  write_output<Eigen::Matrix<double, Dynamic, Dynamic>>(
    "output/test_extend_matrices_body_B.oc", get_output(ssB, B));

  unsigned int N = 9; // horizon length
  Eigen::Tensor<double, 3> S; //tensor to multiply state
  Eigen::Tensor<double, 4> U; //tensor to multiply input
  gecko::tools::extend_matrices(S, U, N, &A, &B);

  std::stringstream ssAa;
  write_output<Eigen::Matrix<double, Dynamic, Dynamic>>(
    "output/test_extend_matrices_body_A_after_extend_matrices.oc", get_output(ssAa, A));
  std::stringstream ssBa;
  write_output<Eigen::Matrix<double, Dynamic, Dynamic>>(
    "output/test_extend_matrices_body_B_after_extend_matrices.oc", get_output(ssBa, B));

  assert(S.dimension(0) == N);
  assert(S.dimension(1) == a0);
  assert(S.dimension(2) == a1);
  assert(U.dimension(0) == b1);
  assert(U.dimension(1) == N);
  assert(U.dimension(2) == N);
  assert(U.dimension(3) == b0);

  std::stringstream ssS;
  write_output<Eigen::Tensor<double, 3>>(
    "output/test_extend_matrices_body_S.oc", get_output(ssS, S));
  std::stringstream ssU;
  write_output<Eigen::Tensor<double, 4>>(
    "output/test_extend_matrices_body_U.oc", get_output(ssU, U));
}

BOOST_AUTO_TEST_SUITE_END()
