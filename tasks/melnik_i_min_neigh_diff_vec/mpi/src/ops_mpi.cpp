#include "melnik_i_min_neigh_diff_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "melnik_i_min_neigh_diff_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace melnik_i_min_neigh_diff_vec {

MelnikIMinNeighDiffVecMPI::MelnikIMinNeighDiffVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(-1, -1);
}

bool MelnikIMinNeighDiffVecMPI::ValidationImpl() {
  return !GetInput().empty() && GetInput().size() > 1;
}

bool MelnikIMinNeighDiffVecMPI::PreProcessingImpl() {
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

bool MelnikIMinNeighDiffVecMPI::RunImpl() {
  return true; // TODO rework in future
}

bool MelnikIMinNeighDiffVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace melnik_i_min_neigh_diff_vec
