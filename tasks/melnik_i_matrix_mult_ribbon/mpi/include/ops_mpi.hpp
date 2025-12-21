#pragma once

#include <cstddef>
#include <vector>

#include "melnik_i_matrix_mult_ribbon/common/include/common.hpp"
#include "task/include/task.hpp"

namespace melnik_i_matrix_mult_ribbon {

class MelnikIMatrixMultRibbonMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit MelnikIMatrixMultRibbonMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Validate input only on rank 0
  bool ValidateOnRoot();
  // Checks if all rows have the same length
  static bool HasUniformRowWidth(const std::vector<std::vector<double>> &matrix, std::size_t expected_width);

  // Broadcast matrix dimensions
  void ShareSizes();
  // Broadcast matrix B to all processes
  void ShareMatrixB();
  // Split A by rows across processes
  std::size_t ScatterRows(std::vector<double> &local_a, std::vector<int> &rows_per_rank);
  // Compute the local block of result matrix
  void MultiplyLocal(const std::vector<double> &local_a, std::vector<double> &local_c) const;
  // Gather local_c from all ranks into the final matrix
  void GatherAndDistribute(const std::vector<int> &rows_per_rank, const std::vector<double> &local_c);

  std::vector<std::vector<double>> matrix_A_;
  std::vector<std::vector<double>> matrix_B_;

  // Current process rank and total number of processes
  // set in the constructor
  int proc_rank_ = 0;
  int proc_num_ = 1;

  std::size_t rows_a_ = 0;
  std::size_t cols_a_ = 0;
  std::size_t rows_b_ = 0;
  std::size_t cols_b_ = 0;

  // keep matrices "flat" to send/receive them efficiently
  std::vector<double> flat_a_;
  std::vector<double> flat_b_transposed_;
  std::vector<double> flat_c_;
};

}  // namespace melnik_i_matrix_mult_ribbon
