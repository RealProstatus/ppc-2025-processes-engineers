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
    if (GetInput().size() < 2)
  {
    return false;
  }

  for (auto it = GetInput().begin(); it != GetInput().end(); ++it)
  {
    if (!std::isfinite(*it))
    {
      return false;
    }
  }

  return true;
}

bool MelnikIMinNeighDiffVecMPI::PreProcessingImpl() {
  return true;
}

bool MelnikIMinNeighDiffVecMPI::RunImpl() {
  return true; // TODO rework in future
}

bool MelnikIMinNeighDiffVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace melnik_i_min_neigh_diff_vec
