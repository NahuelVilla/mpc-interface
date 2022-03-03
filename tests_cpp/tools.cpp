#include <iostream>

// Boost Headers
#include <boost/test/unit_test.hpp>

// Project headers
#include <qp_formulations/tools.hh>


BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)
  
BOOST_AUTO_TEST_CASE(test_extend_matrices)
{
  Eigen::MatrixXd A(3,3);
  A.row(0) << 1. , 0.1 ,0.005;
  A.row(1) << 0. ,   1. ,0.1;
  A.row(2) << 0.  ,  0. ,   1.;
  std::cout << A << std::endl;

  Eigen::MatrixXd B(3,1);
  B.row(0) << 0.00016667;
  B.row(1) << 0.005;
  B.row(2) << 0.1;
  std::cout << B << std::endl;

  Eigen::MatrixXd S;
  Eigen::Tensor<double, 3> U;

  gecko::tools::extend_matrices(S,U,9,A,B);
  
}


BOOST_AUTO_TEST_SUITE_END()
