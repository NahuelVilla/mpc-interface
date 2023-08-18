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

#include <mpc-interface/combinations.hh>

namespace nimbleone {
namespace mpc {

class DomainVariable
{
public:
  /// Constructor
  DomainVariable();

  void init(
    std::vector<std::string> & names,
    std::vector<int> & sizes,
    std::vector<std::string> & axes
  );

  void identify_domain(std::vector<std::string> & names);
  void set_sizes(std::vector<std::string> & names, std::vector<int> & sizes);
  void update_sizes(std::map<std::string, double> & kargs);
  void make_definitions();
  void define_output(
    std::string & name,
    std::map<std::string, int> & combination
  );
  void update_definitions();

private:
  std::vector<std::string> names_;
  std::vector<int> sizes_;
  std::vector<std::string> axes_;
  std::map<std::string, int> domain_ID;
  std::map<std::string, int> domain;
  std::map<std::string, int> all_variables;
  std::map<std::string, LineCombo> definitions;
  std::vector<std::string> outputs;
};

/// *{
/// \class ExtendenSystem
/// This class works a dynamics of the form:
/// \$ x = S*x0 + U*u \$
/// where
/// S: is a 3d tensor of shape \$ [N,n,n] \$.
/// U: is a 4d tensor of shpae \$ [m,N,N,n] \$.
/// with \$ N\$ the horizon length, \$ n\$ the number of states,
/// \$ m \$ number of inputs and \$ p_u \$ the number of ctions predicted
/// for each input.
/// *}

class ControlSystem {
public:
  ControlSystem(
    Eigen::MatrixXd *A, Eigen::MatrixXd *B
  );

  void init(
    std::vector<std::string> &input_names,
    std::vector<std::string> &state_names,
    Eigen::MatrixXd *A, Eigen::MatrixXd *B,
    std::vector<std::string> &axes
  );

  /// getters
  std::vector<std::string> & get_input_names();
  std::vector<std::string> & get_state_names();
  Eigen::MatrixXd * get_matrix_A();
  Eigen::MatrixXd * get_matrix_B();
  std::vector<std::string> & get_axes();

private:
  /// Store the names of the inputs
  std::vector<std::string> input_names_;

  /// Store the names of the states
  std::vector<std::string> state_names_;

  /// A:
  Eigen::MatrixXd A_;

  /// B
  Eigen::MatrixXd B_;

  std::vector<std::string> axes_;
};

class ExtendedSystem {
public:
  /// Constructor
  ExtendedSystem();

  void init(
    std::vector<std::string> &input_names,
    std::vector<std::string> &state_names,
    std::string &state_vector_name,
    std::vector<std::string> &axes
  );

  void init_from_control_system(
    ControlSystem * control_system,
    std::string state_vector_name,
    unsigned int horizon_length
  );

  void identify_domain(std::vector<std::string> &input_name,
                       std::vector<std::string> &state_names);

  /// getters
  Eigen::Tensor<double, 3> & get_S();
  Eigen::Tensor<double, 4> & get_U();
  int get_horizon_length();

   /// TODO ? Change the name of set_sizes
  void set_sizes();

  /// TODO:
  void update_sizes();

private:
  /// Store the list of axis.
  std::vector<std::string> axis_;

  /// Store the names of the inputs
  std::vector<std::string> input_names_;

  /// Store the names of the states
  std::vector<std::string> state_names_;

  /// State vector name
  std::string state_vector_name_;

  /// Matrices
  /// State related matrix
  Eigen::Tensor<double, 3> S_;

  /// Command related matrix
  Eigen::Tensor<double, 4> U_;

  /// Domain ID
  std::map<std::string, int> domain_ID_;

  /// State ID
  std::map<std::string, int> state_ID_;

  /// All variables
  std::map<std::string, int> all_variables_;

  /// List of axis in the model
  std::vector<std::string> axes_;

  /// Horizon Length
  int horizon_length_;
};

}  // namespace mpc
}  // namespace nimbleone
