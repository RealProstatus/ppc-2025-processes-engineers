#include "melnik_i_min_neigh_diff_vec/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "melnik_i_min_neigh_diff_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace melnik_i_min_neigh_diff_vec {

MelnikIMinNeighDiffVecSEQ::MelnikIMinNeighDiffVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(-1, -1);
}

bool MelnikIMinNeighDiffVecSEQ::ValidationImpl() {
  return !GetInput().empty() && GetInput().size() > 1;
}

bool MelnikIMinNeighDiffVecSEQ::PreProcessingImpl() {
  // move this into ValidationImpl
  for (int i = 0; i < GetInput().size(); i++)
  {
    if (!std::isfinite(GetInput()[i]))
    {
      return false;
    }
  }
  
  return true;
}

bool MelnikIMinNeighDiffVecSEQ::RunImpl() {
  const std::vector<double>& inputPtr = GetInput();

  int minIdx = 0;
  double minDiff = std::abs(inputPtr[1] - inputPtr[0]);
  double currDiff;
  
  for (int i = 1; i < inputPtr.size() - 1; i++)
  {
    currDiff = std::abs(inputPtr[i] - inputPtr[i+1]);
    if (currDiff < minDiff)
    {
      minDiff = currDiff;
      minIdx = i;
    }
  }

  GetOutput() = std::make_tuple(minIdx, minIdx + 1);
  return true;
}

bool MelnikIMinNeighDiffVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace melnik_i_min_neigh_diff_vec
