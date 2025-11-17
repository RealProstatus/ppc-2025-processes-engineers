#pragma once

#include <limits>
#include <vector>

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

  struct Result {
    int diff = std::numeric_limits<int>::max();
    int index = -1;
  };

  void ScatterData(std::vector<int> &local_data, const std::vector<int> &counts, const std::vector<int> &displs,
                   int rank);
  void ComputeLocalMin(Result &local_res, const std::vector<int> &local_data, int local_size, int local_displ) const;
  void HandleBoundaryDiffs(Result &local_res, int local_size, const std::vector<int> &local_data, int local_displ,
                           int rank, int comm_size) const;
  void ReduceAndBroadcastResult(Result &global_res, const Result &local_res) const;
};

}  // namespace melnik_i_min_neigh_diff_vec
