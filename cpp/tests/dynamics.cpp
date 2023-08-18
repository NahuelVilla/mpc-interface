#include <iostream>
#include <cassert>

// Boost Headers
#include <boost/test/unit_test.hpp>

// Project headers
#include <mpc-interface/dynamics.hh>
#include <mpc-interface/tools.hh>

// the Agent helps us run tests
#include "testAgent.h"

using namespace Eigen;

const int A_NUM_ROWS = 8;
const int A_NUM_COLUMNS = 8;
const int B_NUM_ROWS = 8;
const int B_NUM_COLUMNS = 6;

void update_matrices_for_control_system(nimbleone::mpc::ControlSystem * cnt_sys, double factor);
void update_matrices_for_extended_system(nimbleone::mpc::ControlSystem * cnt_sys, double factor, nimbleone::mpc::ExtendedSystem * ext_sys);

struct dynamics_fixture
{
  dynamics_fixture() :
    // Settings for control systems
    inputs({"u0", "u1", "u2", "u3", "u4", "u5"}),
    states({"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"}),
    A(A_NUM_ROWS, A_NUM_COLUMNS), B(B_NUM_ROWS, B_NUM_COLUMNS),
    axes({"_x", "_y", "_z", "_a", "_b"}),
    cnt_sys1(&A, &B),
    cnt_sys2(&A, &B),
    cnt_sys3(&A, &B),
    // Setting extended control systems
    horizon_length(20)
  {
    // Settings for control systems
    A.setZero(A_NUM_ROWS, A_NUM_COLUMNS);
    for(int i = 0; i < A_NUM_COLUMNS; i++)
    {
      A(A_NUM_ROWS - 1, i) = 1.0;
    }
    A.block(0, 1, A_NUM_ROWS - 1, A_NUM_COLUMNS - 1).setIdentity();
    //
    for(int i = 0; i < B_NUM_ROWS; i++)
    {
      for(int j = 0; j < B_NUM_COLUMNS; j++)
      {
        B(i, j) = 1.0;
      }
    }

    std::stringstream ssA0;
    write_output<Eigen::Matrix<double, Dynamic, Dynamic>>(
      "output/test_control_system_A0.oc", get_output(ssA0, A));
    std::stringstream ssB0;
    write_output<Eigen::Matrix<double, Dynamic, Dynamic>>(
      "output/test_control_system_B0.oc", get_output(ssB0, B));

    cnt_sys1.init(inputs, states, &A, &B, axes);
    cnt_sys2.init(inputs, states, &A, &B, axes);
    cnt_sys3.init(inputs, states, &A, &B, axes);

    // Setting extended control systems
    ext_sys1.init_from_control_system(&cnt_sys1, "x", horizon_length);
    ext_sys2.init_from_control_system(&cnt_sys2, "e", horizon_length);
    ext_sys3.init_from_control_system(&cnt_sys3, "r", horizon_length);

    std::stringstream ssA1;
    write_output<Eigen::Matrix<double, Dynamic, Dynamic>>(
      "output/test_control_system_A1.oc", get_output(ssA1, *(cnt_sys1.get_matrix_A())));
    std::stringstream ssB1;
    write_output<Eigen::Matrix<double, Dynamic, Dynamic>>(
      "output/test_control_system_B1.oc", get_output(ssB1, *(cnt_sys1.get_matrix_B())));

  }

  ~dynamics_fixture()
  {

  }

  // Settings for control systems
  std::vector<std::string> inputs;
  std::vector<std::string> states;
  Eigen::MatrixXd A;
  Eigen::MatrixXd B;
  std::vector<std::string> axes;
  //
  nimbleone::mpc::ControlSystem cnt_sys1;
  nimbleone::mpc::ControlSystem cnt_sys2;
  nimbleone::mpc::ControlSystem cnt_sys3;

  // Setting extended control systems
  int horizon_length;
  nimbleone::mpc::ExtendedSystem ext_sys1;
  nimbleone::mpc::ExtendedSystem ext_sys2;
  nimbleone::mpc::ExtendedSystem ext_sys3;

        // # #### Settings for single variable
        // def in_this_way(singVar, **kargs):
        //     if isinstance(kargs["new_sizes"], Iterable):
        //         sizes = kargs["new_sizes"]
        //     else:
        //         sizes = [kargs["new_sizes"]]
        //     singVar.domain.update(
        //         {var: sizes[ID] for var, ID in singVar.domain_ID.items()}
        //     )

        // domVar1 = dy.DomainVariable(
        //     "non_lin",
        //     20,
        //     ["_x", "_y"],
        //     time_variant=True,
        //     how_to_update_size=in_this_way,
        // )
        // domVar2 = dy.DomainVariable(
        //     "n", 20, ["_x", "_y"], time_variant=False, how_to_update_size=in_this_way
        // )
        // domVar3 = dy.DomainVariable(["n", "H"], [20, 120], ["_x", "_y"])
        // domVar4 = dy.DomainVariable("o", [20])
        // domVar5 = dy.DomainVariable("o", [20], 5)

        // self.domVar1 = domVar1
        // self.domVar2 = domVar2o one
};

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_FIXTURE_TEST_CASE(test_control_system, dynamics_fixture) {
  assert(cnt_sys1.get_state_names() == states);
  assert(cnt_sys2.get_input_names() == inputs);

  std::stringstream ssBb;
  write_output<Eigen::Matrix<double, Dynamic, Dynamic>>(
    "output/test_control_system_B_before_control.oc", get_output(ssBb, *(cnt_sys1.get_matrix_B())));

  double factor = 3.0;
  update_matrices_for_control_system(&cnt_sys1, factor);
  update_matrices_for_control_system(&cnt_sys2, factor);
  update_matrices_for_control_system(&cnt_sys3, factor);

  Eigen::MatrixXd correct_new_B_1(B_NUM_ROWS, B_NUM_COLUMNS);
  for(int i = 0; i < B_NUM_ROWS; i++)
  {
    for(int j = 0; j < B_NUM_COLUMNS; j++)
    {
      double val = 3.0;
      if(i == B_NUM_ROWS - 1) val = 24.0;
      correct_new_B_1(i, j) = val;
    }
  }

  std::stringstream ssA;
  write_output<Eigen::Matrix<double, Dynamic, Dynamic>>(
    "output/test_control_system_A_after_control.oc", get_output(ssA, *(cnt_sys1.get_matrix_A())));
  std::stringstream ssB;
  write_output<Eigen::Matrix<double, Dynamic, Dynamic>>(
    "output/test_control_system_B_after_control.oc", get_output(ssB, *(cnt_sys1.get_matrix_B())));
  std::stringstream ss_correct_new_B_1;
  write_output<Eigen::Matrix<double, Dynamic, Dynamic>>(
    "output/test_control_system_correct_new_B_1.oc", get_output(ss_correct_new_B_1, correct_new_B_1));

//   assert(*(cnt_sys1.get_matrix_B()) == correct_new_B_1);

//   assert(cnt_sys2.B == self.B).all())
//   assert(cnt_sys3.B == self.B).all())

//   LIP = dy.ControlSystem.from_name("J->CCC", ["_x", "_y"], tau=0.1, omega=3.5)
//   assert(LIP.parameters["tau"], 0.1)
//   assert(LIP.parameters["omega"], 3.5)
//   correctA = use.get_system_matrices("J->CCC")[0](tau=0.1, omega=3.5)
//   correctB = use.get_system_matrices("J->CCC")[1](tau=0.1, omega=3.5)

//   assert(LIP.A == correctA).all())
//   assert(LIP.B == correctB).all()) double factor
}

BOOST_AUTO_TEST_SUITE_END()

void update_matrices_for_control_system(nimbleone::mpc::ControlSystem * cnt_sys, double factor)
{
    Eigen::MatrixXd newB =
      *(cnt_sys->get_matrix_A()) * *(cnt_sys->get_matrix_B()) * factor;
    *(cnt_sys->get_matrix_B()) = newB;
}

void update_matrices_for_extended_system(nimbleone::mpc::ControlSystem * cnt_sys, double factor, nimbleone::mpc::ExtendedSystem * ext_sys)
{
  update_matrices_for_control_system(cnt_sys, factor);

  nimbleone::mpc::extend_matrices(ext_sys->get_S(), ext_sys->get_U(), ext_sys->get_horizon_length(),
    cnt_sys->get_matrix_A(), cnt_sys->get_matrix_B());
}
