///
/// Gecko - Tools
/// Author: Olivier Stasse
/// Copyright: LAAS-CNRS
/// Date: 2022
///
/// Author2: David Bellis
/// Copyright2: Nimble One
/// Date2: Aug 2023
///
///

#ifndef QP_FORMULATIONS_TOOLS_H_
#define QP_FORMULATIONS_TOOLS_H_

// Standard C++ include
#include <iostream>
#include <iomanip>

// This repository includes
#include <mpc-interface/tools.hh>

namespace gecko {
namespace tools {

using namespace Eigen;

template <typename T>
using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar, int rank, typename sizeType>
auto Tensor_to_Matrix(const Eigen::Tensor<Scalar, rank> &tensor,
                      const sizeType rows, const sizeType cols) {
  return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols);
}

template <typename Scalar, typename... Dims>
auto Matrix_to_Tensor(const MatrixType<Scalar> &matrix, Dims... dims) {
  constexpr int rank = sizeof...(Dims);
  return Eigen::TensorMap<Eigen::Tensor<const Scalar, rank>>(matrix.data(),
                                                             {dims...});
}

void extend_matrices(Eigen::Tensor<double, 3> &S, Eigen::Tensor<double, 4> &U,
                     unsigned int N, Eigen::MatrixXd *A, Eigen::MatrixXd *B) {
  Index n = B->rows();
  Index m = B->cols();

  MatrixXd s(n * N, n);
  s.setZero();
  Tensor<double, 3> u(n * N, N, m);
  u.setZero();

  s.block(0, 0, n, n) = *A;
  for (Index ind_n = 0; ind_n < n; ind_n++)
    for (Index ind_m = 0; ind_m < m; ind_m++)
      u(ind_n, 0, ind_m) = (*B)(ind_n, ind_m);

  for (Index i = 1; i < N; i++) {
    for (Index j = 0; j < m; j++) {
      Eigen::array<Index, 3> offsets = {(n * (i - 1)), 0, j};
      Eigen::array<Index, 3> extents = {n, i, 1};
      Eigen::Tensor<double, 3> sub_u = u.slice(offsets, extents);
      Eigen::MatrixXd sub_u_m = Tensor_to_Matrix(sub_u, n, i);

      Eigen::MatrixXd sub_u_adot = *A * sub_u_m;
      Eigen::array<Index, 3> offsets_concat = {n * i, 0, j};
      Eigen::array<Index, 3> extents_concat = {n, i, 1};

      u.slice(offsets_concat, extents_concat) =
          Matrix_to_Tensor(sub_u_adot, n, i, 1);
    }

    Eigen::array<Index, 3> offsets_B = {n * i, i, 0};
    Eigen::array<Index, 3> extents_B = {n, 1, m};

    u.slice(offsets_B, extents_B) = Matrix_to_Tensor(*B, n, 1, m);

    Eigen::array<Index, 3> offsets_disp = {n * i, 0, 0};
    Eigen::array<Index, 3> extents_disp = {n, i, m};

    Eigen::Tensor<double, 3> u_disp = u.slice(offsets_disp, extents_disp);

    s.block(n * i, 0, n, n) = *A * s.block(n * (i - 1), 0, n, n);
  }

  S.resize(N, n, n);
  for (Index i = 0; i < N; i++) {
    Eigen::array<Index, 3> offsets_S = {i, 0, 0};
    Eigen::array<Index, 3> extents_S = {1, n, n};

    MatrixXd sb = s.block(n * i, 0, n, n);
    MatrixXd sb_t = sb.transpose();
    S.slice(offsets_S, extents_S) = Matrix_to_Tensor(sb_t, 1, n, n);
  }

  U.resize(m, N, N, n);
  for (Index j = 0; j < m; j++)
    for (Index i = 0, lU_i = 0; i < n * N; i += n, lU_i++)
      for (Index k = 0; k < N; k++)
        for (Index l = 0; l < n; l++) U(j, lU_i, k, l) = u(i + l, k, j);
}

void update_step_matrices(std::shared_ptr<ExtendedSystem> /*shr_ext_sys*/,
                          std::map<std::string, double> & /*kargs*/) {
  /*This function needs
    count : int, representing the current time sample number
    and one of the following:
      step_times : ndarray or list with next step times.
    or
      regular_time : int, to produce steps regularly
  */
/*
  auto matrices = shr_ext_sys->get_matrices();
  auto S = std::get<1>(matrices);
  unsigned int N = S.dimension(0);

  double count = 0;
  if (auto search = kargs.find("count"); search != kargs.end())
    count = search->second;


    if "step_times" in kargs.keys():
        step_times = kargs["step_times"]
        regular_time = None

    elif "regular_time" in kargs.keys():
        regular_time = kargs["regular_time"]
        step_times = None

    else:
        raise KeyError(
            "This funtion needs either 'step_times' or "
            + "'regular_time', but the kargs "
            + "introduced are {}".format(kargs.keys())
        )

    if "w_phase" in kargs.keys():
        walking_phase = kargs["w_phase"]
    else:
        walking_phase = 0

    U = plan_steps(N, count, step_times, regular_time, walking_phase)
    extSyst.matrices[0] = U[:, :, None]
    */

}

void get_system_matrices(std::string & /*system*/) {}

}  // namespace tools
}  // namespace gecko
#endif
