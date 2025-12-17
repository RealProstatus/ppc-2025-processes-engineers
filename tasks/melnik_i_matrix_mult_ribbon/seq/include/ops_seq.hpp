#pragma once

#include "melnik_i_matrix_mult_ribbon/common/include/common.hpp"
#include "task/include/task.hpp"

namespace melnik_i_matrix_mult_ribbon {

class MelnikIMatrixMultRibbonSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MelnikIMatrixMultRibbonSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace melnik_i_matrix_mult_ribbon
