#pragma once

#include "melnik_i_gauss_block_part/common/include/common.hpp"
#include "task/include/task.hpp"

namespace melnik_i_gauss_block_part {

class MelnikIGaussBlockPartSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MelnikIGaussBlockPartSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ApplyGaussian(const InType &input, OutType &output);
};

}  // namespace melnik_i_gauss_block_part
