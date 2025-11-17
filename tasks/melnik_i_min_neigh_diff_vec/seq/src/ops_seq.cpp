#include "melnik_i_min_neigh_diff_vec/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdlib>
#include <ranges>
#include <tuple>
#include <vector>

#include "melnik_i_min_neigh_diff_vec/common/include/common.hpp"

namespace melnik_i_min_neigh_diff_vec {

MelnikIMinNeighDiffVecSEQ::MelnikIMinNeighDiffVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(-1, -1);
}

bool MelnikIMinNeighDiffVecSEQ::ValidationImpl() {
  return GetInput().size() >= 2;
}

bool MelnikIMinNeighDiffVecSEQ::PreProcessingImpl() {
  return true;
}

bool MelnikIMinNeighDiffVecSEQ::RunImpl() {
  const auto &v = GetInput();

  int best_idx = 0;
  int best_diff = std::abs(v[1] - v[0]);
  for (size_t i = 1; i + 1 < v.size(); i++) {
    int delta = std::abs(v[i + 1] - v[i]);
    if (delta < best_diff || (delta == best_diff && static_cast<int>(i) < best_idx)) {
      best_diff = delta;
      best_idx = static_cast<int>(i);
    }
  }

  GetOutput() = std::make_tuple(best_idx, best_idx + 1);
  return true;
}

bool MelnikIMinNeighDiffVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace melnik_i_min_neigh_diff_vec
