#ifndef TEST_AGENT_H
#define TEST_AGENT_H

// basic file operations
#include <iostream>
#include <fstream>

template<typename T>
void write_output(const char * filename, std::stringstream & ss)
{
    std::ofstream output_file;
    output_file.open (filename);
    output_file << ss.str();
    output_file.close();
}

// output Eigen data structures
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
using namespace Eigen;

#include <iomanip>
#define REAL_PRECISION 16

inline std::stringstream & get_output(std::stringstream & ss, Eigen::Matrix<double, Dynamic, Dynamic> &mat) {
  ss << "[" << std::endl;
  Index R = mat.rows();
  Index C = mat.cols();
  for (Index r = 0; r < R; r++) {
    ss << "  [";
    for (Index c = 0; c < C; c++) {
      ss << std::fixed << std::setprecision(REAL_PRECISION) << mat(r, c);
      if (c != C - 1) ss << ",";
    }
    ss << "]" << std::endl ;
  }
  ss << "]" << std::endl;
  return ss;
}

inline std::stringstream & get_output(std::stringstream & ss, Eigen::Tensor<double, 3> &aT) {
  ss << "(" << aT.dimension(0) << "," << aT.dimension(1) << ","
            << aT.dimension(2) << ")="
            << std::endl << "[" << std::endl;
  for (Index i0 = 0; i0 < aT.dimension(0); i0++) {
    ss << "  [" << std::endl;
    for (Index i1 = 0; i1 < aT.dimension(1); i1++) {
      ss << "    [";
      for (Index i2 = 0; i2 < aT.dimension(2); i2++) {
        ss << std::fixed << std::setprecision(REAL_PRECISION) << aT(i0, i1, i2);
        if (i2 != aT.dimension(2) - 1) ss << ",";
      }
      ss << "]";
      if (i1 != aT.dimension(1) - 1) ss << std::endl;
    }
    ss << std::endl << "  ]";
    if (i0 != aT.dimension(0) - 1) ss << std::endl;
  }
  ss << std::endl << "]" << std::endl;
  return ss;
}

inline std::stringstream & get_output(std::stringstream & ss, Eigen::Tensor<double, 4> &aT) {
  ss << "(" << aT.dimension(0) << "," << aT.dimension(1) << ","
            << aT.dimension(2) << "," << aT.dimension(3) << ")="
            << std::endl << "[" << std::endl;
  for (Index i0 = 0; i0 < aT.dimension(0); i0++) {
    ss << "  [" << std::endl;
    for (Index i1 = 0; i1 < aT.dimension(1); i1++) {
      ss << "    [" << std::endl;
      for (Index i2 = 0; i2 < aT.dimension(2); i2++) {
        if (aT.dimension(3) > 1) ss << "      [";
        for (Index i3 = 0; i3 < aT.dimension(3); i3++) {
          ss << std::fixed << std::setprecision(REAL_PRECISION) << aT(i0, i1, i2, i3);
          if (i3 != aT.dimension(3) - 1) ss << ",";
        }
        if (aT.dimension(3) > 1) ss << "]";
        if (i2 != aT.dimension(2) - 1) ss << std::endl;
      }
      ss << std::endl << "    ]";
      if (i1 != aT.dimension(1) - 1) ss << std::endl;
    }
    ss << std::endl << "  ]";
  }
  ss << std::endl << "]" << std::endl;
  return ss;
}

#endif
