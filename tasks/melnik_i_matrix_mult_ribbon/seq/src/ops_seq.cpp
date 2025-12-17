#include "melnik_i_matrix_mult_ribbon/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "melnik_i_matrix_mult_ribbon/common/include/common.hpp"

namespace melnik_i_matrix_mult_ribbon {

MelnikIMatrixMultRibbonSEQ::MelnikIMatrixMultRibbonSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<std::vector<double>>();
}

bool MelnikIMatrixMultRibbonSEQ::ValidationImpl() {
  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  return !matrix_a.empty() && !matrix_b.empty() && matrix_a[0].size() == matrix_b.size();
}

bool MelnikIMatrixMultRibbonSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool MelnikIMatrixMultRibbonSEQ::RunImpl() {
  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();
  size_t cols_b = matrix_b[0].size();

  auto &output = GetOutput();
  output = std::vector<std::vector<double>>(rows_a, std::vector<double>(cols_b, 0.0));

  for (size_t i = 0; i < rows_a; i++) {
    for (size_t k = 0; k < cols_a; k++) {
      double aik = matrix_a[i][k];
      for (size_t j = 0; j < cols_b; j++) {
        output[i][j] += aik * matrix_b[k][j];
      }
    }
  }

  return true;
}

bool MelnikIMatrixMultRibbonSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace melnik_i_matrix_mult_ribbon
