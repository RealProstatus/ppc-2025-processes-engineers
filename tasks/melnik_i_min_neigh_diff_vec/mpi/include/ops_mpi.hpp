#pragma once

#include "melnik_i_min_neigh_diff_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace melnik_i_min_neigh_diff_vec {

class MelnikIMinNeighDiffVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MelnikIMinNeighDiffVecMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace melnik_i_min_neigh_diff_vec
