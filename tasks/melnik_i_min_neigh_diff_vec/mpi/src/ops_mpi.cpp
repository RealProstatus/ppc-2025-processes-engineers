#include "melnik_i_min_neigh_diff_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
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

bool MelnikIMinNeighDiffVecMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int comm_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  const auto &global_input = GetInput();
  int global_size = 0;

  if (rank == 0) {
    global_size = static_cast<int>(global_input.size());
  }
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (global_size < 2) {
    return false;
  }

  // Prepare for data distribution: counts and displacements
  std::vector<int> counts(comm_size);
  std::vector<int> displs(comm_size);
  int local_size = 0;

  if (rank == 0) {
    // Balanced distribution: base size + remainder for first processes
    int base = global_size / comm_size;
    int rem = global_size % comm_size;
    int offset = 0;
    for (int i = 0; i < comm_size; ++i) {
      counts[i] = base + (i < rem ? 1 : 0);
      displs[i] = offset;
      offset += counts[i];
    }
  }

  // Scatter local sizes to each process
  MPI_Scatter(counts.data(), 1, MPI_INT, &local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Allocate local data buffer
  std::vector<int> local_data(local_size);

  // Scatter the actual data chunks
  MPI_Scatterv(rank == 0 ? global_input.data() : nullptr, counts.data(), displs.data(), MPI_INT, local_data.data(),
               local_size, MPI_INT, 0, MPI_COMM_WORLD);

  // Compute local displacement
  int local_displ = 0;
  MPI_Scan(&local_size, &local_displ, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  local_displ -= local_size;

  // Structure to hold min diff and its global index
  struct Result {
    int diff = std::numeric_limits<int>::max();
    int index = -1;
  } local_res, global_res;

  // Local computation: find min diff in this chunk
  if (local_size >= 2) {
    for (int i = 0; i < local_size - 1; ++i) {
      int curr_diff = std::abs(local_data[i + 1] - local_data[i]);
      // Update if smaller diff, or equal diff but smaller index (for stability)
      if (curr_diff < local_res.diff) {
        local_res.diff = curr_diff;
        local_res.index = local_displ + i;
      } else if (curr_diff == local_res.diff && (local_displ + i) < local_res.index) {
        local_res.index = local_displ + i;
      }
    }
  }

  // Handle boundary pairs between processes if multiple processes
  if (comm_size > 1) {
    // Get boundary values (or 0 if empty chunk)
    int left_boundary = local_size > 0 ? local_data.front() : 0;
    int right_boundary = local_size > 0 ? local_data.back() : 0;

    int recv_from_left = 0;
    int recv_from_right = 0;

    // Prepare non-blocking requests for efficiency
    std::array<MPI_Request, 4> requests;
    int req_count = 0;

    // Exchange with left neighbor
    if (rank > 0) {
      // Send my left to left neighbor (their right)
      MPI_Isend(&left_boundary, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
      // Receive left neighbor's right
      MPI_Irecv(&recv_from_left, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Exchange with right neighbor
    if (rank < comm_size - 1) {
      // Receive right neighbor's left
      MPI_Irecv(&recv_from_right, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
      // Send my right to right neighbor (their left)
      MPI_Isend(&right_boundary, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Wait for all exchanges to complete
    if (req_count > 0) {
      MPI_Waitall(req_count, requests.data(), MPI_STATUSES_IGNORE);
    }

    // Check left boundary diff
    if (rank > 0 && local_size > 0) {
      int boundary_diff = std::abs(left_boundary - recv_from_left);
      int boundary_idx = local_displ - 1;
      if (boundary_diff < local_res.diff) {
        local_res.diff = boundary_diff;
        local_res.index = boundary_idx;
      } else if (boundary_diff == local_res.diff && boundary_idx < local_res.index) {
        local_res.index = boundary_idx;
      }
    }

    // Check right boundary diff
    if (rank < comm_size - 1 && local_size > 0) {
      int boundary_diff = std::abs(recv_from_right - right_boundary);
      int boundary_idx = local_displ + local_size - 1;
      if (boundary_diff < local_res.diff) {
        local_res.diff = boundary_diff;
        local_res.index = boundary_idx;
      } else if (boundary_diff == local_res.diff && boundary_idx < local_res.index) {
        local_res.index = boundary_idx;
      }
    }
  }

  // Global reduction: find overall min using MPI_MINLOC for {diff, index}
  MPI_Reduce(&local_res, &global_res, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

  // Broadcast the global result to all processes
  MPI_Bcast(&global_res, 1, MPI_2INT, 0, MPI_COMM_WORLD);

  // Set output if valid
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
