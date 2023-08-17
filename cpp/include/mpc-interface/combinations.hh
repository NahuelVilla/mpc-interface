///
/// Gecko - Tools
/// Author: David Bellis
/// Copyright: Nimble One
/// Date: Aug 2023
///

#ifndef GECKO_COMBINATIONS_H_
#define GECKO_COMBINATIONS_H_

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

namespace gecko {
namespace tools {

class LineCombo
{
public:
  LineCombo(
    std::map<std::string, Eigen::MatrixXd> combination,
    std::vector<std::string> * data,
    bool time_variant,
    std::function<void(
      void ** objects,
      std::map<std::string, double> & kargs
    )> how_to_update
  );
  LineCombo() = default;

  void __figuring_out(
    void ** objects,
    std::map<std::string, double> & kargs
  );

  Eigen::MatrixXd & getitem(std::string variable);
  std::map<std::string, Eigen::MatrixXd> & items();
  std::vector<std::string> & keys();
  std::vector<Eigen::MatrixXd> & values();
  std::string output();
  void update(std::map<std::string, double> & kargs);

public:
  std::map<std::string, Eigen::MatrixXd> combination_;
  std::vector<std::string> *data_;
  bool time_variant_;
  std::function<void(
      void ** objects,
      std::map<std::string, double> & kargs
    )> how_to_update_;

  std::vector<std::string> variables;
  std::vector<Eigen::MatrixXd> matrices;
  std::vector<std::string> _coefficients;
};

}  // namespace tools
}  // namespace gecko
#endif
