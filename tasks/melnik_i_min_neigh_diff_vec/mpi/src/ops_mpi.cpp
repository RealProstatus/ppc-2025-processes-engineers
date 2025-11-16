#include "melnik_i_min_neigh_diff_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <vector>

#include "melnik_i_min_neigh_diff_vec/common/include/common.hpp"

namespace melnik_i_min_neigh_diff_vec {

MelnikIMinNeighDiffVecMPI::MelnikIMinNeighDiffVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(-1, -1);
}

bool MelnikIMinNeighDiffVecMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int is_valid = 1;
  if (rank == 0) {
    const auto &input = GetInput();
    if (input.size() < 2) {
      is_valid = 0;
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid != 0;
}

bool MelnikIMinNeighDiffVecMPI::PreProcessingImpl() {
  return true;
}

void MelnikIMinNeighDiffVecMPI::ScatterData(std::vector<int> &local_data, const std::vector<int> &counts,
                                            const std::vector<int> &displs, int rank) {
  int *sendbuf = (rank == 0) ? GetInput().data() : nullptr;
  MPI_Scatterv(sendbuf, counts.data(), displs.data(), MPI_INT, local_data.data(), static_cast<int>(local_data.size()),
               MPI_INT, 0, MPI_COMM_WORLD);
}

void MelnikIMinNeighDiffVecMPI::ComputeLocalMin(Result &local_res, const std::vector<int> &local_data, int local_size,
                                                int local_displ) {
  local_res.diff = std::numeric_limits<int>::max();
  local_res.index = -1;
  if (local_size < 2) {
    return;
  }
  for (int i = 0; i < local_size - 1; ++i) {
    int curr_diff = std::abs(local_data[i + 1] - local_data[i]);
    if (curr_diff < local_res.diff || (curr_diff == local_res.diff && (local_displ + i) < local_res.index)) {
      local_res.diff = curr_diff;
      local_res.index = local_displ + i;
    }
  }
}

void MelnikIMinNeighDiffVecMPI::UpdateMin(Result &res, int candidate_diff, int candidate_idx) {
  if (candidate_diff < res.diff || (candidate_diff == res.diff && candidate_idx < res.index)) {
    res.diff = candidate_diff;
    res.index = candidate_idx;
  }
}

void MelnikIMinNeighDiffVecMPI::HandleBoundaryDiffs(Result &local_res, int local_size,
                                                    const std::vector<int> &local_data, int local_displ, int rank,
                                                    int comm_size) {
  if (local_size <= 0) {
    return;
  }

  int left_boundary = local_data.front();
  int right_boundary = local_data.back();

  int recv_from_left = 0;
  int recv_from_right = 0;

  std::array<MPI_Request, 4> requests = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  int left_dest = rank > 0 ? rank - 1 : MPI_PROC_NULL;
  int left_src = rank > 0 ? rank - 1 : MPI_PROC_NULL;
  int right_src = rank < comm_size - 1 ? rank + 1 : MPI_PROC_NULL;
  int right_dest = rank < comm_size - 1 ? rank + 1 : MPI_PROC_NULL;

  MPI_Isend(&left_boundary, 1, MPI_INT, left_dest, 0, MPI_COMM_WORLD, &requests[0]);
  MPI_Irecv(&recv_from_left, 1, MPI_INT, left_src, 1, MPI_COMM_WORLD, &requests[1]);
  MPI_Irecv(&recv_from_right, 1, MPI_INT, right_src, 0, MPI_COMM_WORLD, &requests[2]);
  MPI_Isend(&right_boundary, 1, MPI_INT, right_dest, 1, MPI_COMM_WORLD, &requests[3]);

  MPI_Waitall(4, requests.data(), MPI_STATUSES_IGNORE);

  int max_diff = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::max();

  int left_diff = rank > 0 ? std::abs(left_boundary - recv_from_left) : max_diff;
  int left_idx = rank > 0 ? local_displ - 1 : max_idx;
  UpdateMin(local_res, left_diff, left_idx);

  int right_diff = rank < comm_size - 1 ? std::abs(recv_from_right - right_boundary) : max_diff;
  int right_idx = rank < comm_size - 1 ? local_displ + local_size - 1 : max_idx;
  UpdateMin(local_res, right_diff, right_idx);
}

void MelnikIMinNeighDiffVecMPI::ReduceAndBroadcastResult(Result &global_res, const Result &local_res) {
  MPI_Reduce(&local_res, &global_res, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_res, 1, MPI_2INT, 0, MPI_COMM_WORLD);
}

bool MelnikIMinNeighDiffVecMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int comm_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int global_size = rank != 0 ? 0 : static_cast<int>(GetInput().size());
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (global_size < 2) {
    return false;
  }

  std::vector<int> counts(comm_size);
  std::vector<int> displs(comm_size);
  if (rank == 0) {
    int base = global_size / comm_size;
    int rem = global_size % comm_size;
    int offset = 0;
    for (int i = 0; i < comm_size; ++i) {
      counts[i] = base + (i < rem);
      displs[i] = offset;
      offset += counts[i];
    }
  }

  int local_size = 0;
  MPI_Scatter(counts.data(), 1, MPI_INT, &local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_data(local_size);
  ScatterData(local_data, counts, displs, rank);

  int local_displ = 0;
  MPI_Scan(&local_size, &local_displ, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  local_displ -= local_size;

  Result local_res;
  ComputeLocalMin(local_res, local_data, local_size, local_displ);

  if (comm_size > 1) {
    HandleBoundaryDiffs(local_res, local_size, local_data, local_displ, rank, comm_size);
  }

  Result global_res;
  ReduceAndBroadcastResult(global_res, local_res);

  if (global_res.index >= 0) {
    GetOutput() = std::make_tuple(global_res.index, global_res.index + 1);
    return true;
  }
  return false;
}

bool MelnikIMinNeighDiffVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace melnik_i_min_neigh_diff_vec
