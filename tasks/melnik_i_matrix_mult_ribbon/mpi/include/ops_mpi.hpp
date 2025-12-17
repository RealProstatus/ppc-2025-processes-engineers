#pragma once

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

  bool RunSequential();
  void BroadcastMatrixB(std::vector<double> &b_flat, int rows_b, int cols_b);
  void ScatterMatrixA(std::vector<int> &counts, std::vector<int> &displs, std::vector<double> &local_a_flat,
                      int &local_rows, int rows_a, int cols_a, int world_size);
  static void ComputeLocalMultiplication(const std::vector<double> &local_a_flat, const std::vector<double> &b_flat,
                                         std::vector<double> &local_c_flat, int local_rows, int cols_a, int cols_b);
  void GatherMatrixC(std::vector<double> &final_result_flat, const std::vector<int> &counts,
                     const std::vector<int> &displs, const std::vector<double> &local_c_flat, int rows_a, int cols_b,
                     int world_size);
  void ConvertToMatrix(const std::vector<double> &final_result_flat, int rows_a, int cols_b);

  std::vector<std::vector<double>> matrix_A_;
  std::vector<std::vector<double>> matrix_B_;
  int rows_a_ = 0;
  int cols_a_ = 0;
  int rows_b_ = 0;
  int cols_b_ = 0;
};

}  // namespace melnik_i_matrix_mult_ribbon
