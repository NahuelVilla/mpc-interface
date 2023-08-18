///
/// Author: David Bellis
/// Copyright: Nimble One
/// Date: Aug 2023
///
#pragma once

#include <Eigen/Eigen>
#include <map>
#include <vector>
#include <functional>
#include <memory>

#if !EIGEN_HAS_CXX11
#define XSTR(x) STR(x)
#define STR(x) #x
#pragma message "The value of EIGEN_MAX_CPP_VER is " XSTR(EIGEN_MAX_CPP_VER)
#pragma message "The value of EIGEN_COMP_CXXVER is " XSTR(EIGEN_COMP_CXXVER)
#error EIGEN_HAS_CXX11 is required.
#endif

#include <unsupported/Eigen/CXX11/Tensor>

namespace nimbleone {
namespace mpc {

class LineCombo
{
public:
    LineCombo(std::map<std::string, Eigen::MatrixXd> combination);
    LineCombo() = default;

    Eigen::MatrixXd & getitem(std::string variable);
    std::map<std::string, Eigen::MatrixXd> & items();
    std::vector<std::string> & keys();
    std::vector<Eigen::MatrixXd> & values();
    std::string output();

public:
    std::map<std::string, Eigen::MatrixXd> combination_;
    std::vector<std::string> variables;
    std::vector<Eigen::MatrixXd> matrices;
    std::vector<std::string> _coefficients;
};

}  // namespace mpc
}  // namespace nimbleone

