#include "melnik_i_min_neigh_diff_vec/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
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
  const auto &input = GetInput();
  if (input.size() < 2) {
    return false;
  }

  auto iota_range = std::views::iota(size_t{0}, input.size() - 1);
  auto comparator = [&](size_t i, size_t j) {
    int diff_i = std::abs(input[i] - input[i + 1]);
    int diff_j = std::abs(input[j] - input[j + 1]);
    if (diff_i == diff_j) {
      return i < j;
    }
    return diff_i < diff_j;
  };
  auto min_it = std::ranges::min_element(iota_range, comparator);
  int min_idx = static_cast<int>(*min_it);

  GetOutput() = std::make_tuple(min_idx, min_idx + 1);
  return true;
}

bool MelnikIMinNeighDiffVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace melnik_i_min_neigh_diff_vec
