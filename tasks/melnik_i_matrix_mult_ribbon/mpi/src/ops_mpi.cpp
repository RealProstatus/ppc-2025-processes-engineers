#include "melnik_i_matrix_mult_ribbon/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <vector>

#include "melnik_i_matrix_mult_ribbon/common/include/common.hpp"

namespace melnik_i_matrix_mult_ribbon {

MelnikIMatrixMultRibbonMPI::MelnikIMatrixMultRibbonMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = std::vector<std::vector<double>>();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    matrix_A_ = std::get<0>(in);
    matrix_B_ = std::get<1>(in);
  }
}

bool MelnikIMatrixMultRibbonMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    rows_a_ = static_cast<int>(matrix_A_.size());
    cols_a_ = static_cast<int>(matrix_A_[0].size());
    rows_b_ = static_cast<int>(matrix_B_.size());
    cols_b_ = static_cast<int>(matrix_B_[0].size());
  }

  std::array<int, 4> sizes = {rows_a_, cols_a_, rows_b_, cols_b_};
  MPI_Bcast(sizes.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);

  rows_a_ = sizes[0];
  cols_a_ = sizes[1];
  rows_b_ = sizes[2];
  cols_b_ = sizes[3];

  return !(rows_a_ == 0 || rows_b_ == 0 || cols_a_ == 0 || cols_b_ == 0) && cols_a_ == rows_b_;
}

bool MelnikIMatrixMultRibbonMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool MelnikIMatrixMultRibbonMPI::RunImpl() {
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<double> b_flat(static_cast<size_t>(rows_b_) * static_cast<size_t>(cols_b_));
  BroadcastMatrixB(b_flat, rows_b_, cols_b_);

  std::vector<int> counts(world_size);
  std::vector<int> displs(world_size);
  std::vector<double> local_a_flat;
  int local_rows = 0;
  ScatterMatrixA(counts, displs, local_a_flat, local_rows, rows_a_, cols_a_, world_size);

  std::vector<double> local_c_flat(static_cast<size_t>(local_rows) * static_cast<size_t>(cols_b_), 0.0);
  ComputeLocalMultiplication(local_a_flat, b_flat, local_c_flat, local_rows, cols_a_, cols_b_);

  std::vector<double> final_result_flat;
  GatherMatrixC(final_result_flat, counts, displs, local_c_flat, local_rows, rows_a_, cols_b_, world_size);

  if (rank == 0) {
    ConvertToMatrix(final_result_flat, rows_a_, cols_b_);
  }

  return true;
}

void MelnikIMatrixMultRibbonMPI::BroadcastMatrixB(std::vector<double> &b_flat, int rows_b, int cols_b) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    for (int i = 0; i < rows_b; ++i) {
      for (int j = 0; j < cols_b; ++j) {
        b_flat[(static_cast<size_t>(i) * static_cast<size_t>(cols_b)) + static_cast<size_t>(j)] = matrix_B_[i][j];
      }
    }
  }

  MPI_Bcast(b_flat.data(), rows_b * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void MelnikIMatrixMultRibbonMPI::ScatterMatrixA(std::vector<int> &counts, std::vector<int> &displs,
                                                std::vector<double> &local_a_flat, int &local_rows, int rows_a,
                                                int cols_a, int world_size) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int base = rows_a / world_size;
  int remainder = rows_a % world_size;

  int offset = 0;
  for (int i = 0; i < world_size; i++) {
    counts[i] = base + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }

  local_rows = counts[rank];
  local_a_flat.resize(static_cast<size_t>(local_rows) * static_cast<size_t>(cols_a));

  std::vector<double> a_flat;
  if (rank == 0) {
    a_flat.resize(static_cast<size_t>(rows_a) * static_cast<size_t>(cols_a));
    for (int i = 0; i < rows_a; ++i) {
      for (int j = 0; j < cols_a; ++j) {
        a_flat[(static_cast<size_t>(i) * static_cast<size_t>(cols_a)) + static_cast<size_t>(j)] = matrix_A_[i][j];
      }
    }
  }

  std::vector<int> sendcounts(world_size);
  std::vector<int> senddispls(world_size);
  for (int i = 0; i < world_size; ++i) {
    sendcounts[i] = counts[i] * cols_a;
    senddispls[i] = displs[i] * cols_a;
  }

  MPI_Scatterv(rank == 0 ? a_flat.data() : nullptr, sendcounts.data(), senddispls.data(), MPI_DOUBLE,
               local_a_flat.data(), local_rows * cols_a, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void MelnikIMatrixMultRibbonMPI::ComputeLocalMultiplication(const std::vector<double> &local_a_flat,
                                                            const std::vector<double> &b_flat,
                                                            std::vector<double> &local_c_flat, int local_rows,
                                                            int cols_a, int cols_b) {
  for (int i = 0; i < local_rows; ++i) {
    const double *a_row = &local_a_flat[static_cast<size_t>(i) * static_cast<size_t>(cols_a)];
    double *c_row = &local_c_flat[static_cast<size_t>(i) * static_cast<size_t>(cols_b)];

    for (int k = 0; k < cols_a; ++k) {
      double aik = a_row[k];
      const double *b_row = &b_flat[static_cast<size_t>(k) * static_cast<size_t>(cols_b)];

      for (int j = 0; j < cols_b; ++j) {
        c_row[j] += aik * b_row[j];
      }
    }
  }
}

void MelnikIMatrixMultRibbonMPI::GatherMatrixC(std::vector<double> &final_result_flat, const std::vector<int> &counts,
                                               const std::vector<int> &displs, const std::vector<double> &local_c_flat,
                                               int local_rows, int rows_a, int cols_b, int world_size) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> recvcounts(world_size);
  std::vector<int> recvdispls(world_size);
  for (int i = 0; i < world_size; ++i) {
    recvcounts[i] = counts[i] * cols_b;
    recvdispls[i] = displs[i] * cols_b;
  }

  if (rank == 0) {
    final_result_flat.resize(static_cast<size_t>(rows_a) * static_cast<size_t>(cols_b), 0.0);
  }

  MPI_Gatherv(local_c_flat.data(), local_rows * cols_b, MPI_DOUBLE, rank == 0 ? final_result_flat.data() : nullptr,
              recvcounts.data(), recvdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void MelnikIMatrixMultRibbonMPI::ConvertToMatrix(const std::vector<double> &final_result_flat, int rows_a, int cols_b) {
  std::vector<std::vector<double>> result_matrix(rows_a, std::vector<double>(cols_b));

  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      result_matrix[i][j] =
          final_result_flat[(static_cast<size_t>(i) * static_cast<size_t>(cols_b)) + static_cast<size_t>(j)];
    }
  }

  GetOutput() = result_matrix;
}

bool MelnikIMatrixMultRibbonMPI::PostProcessingImpl() {
  return true;
}

}  // namespace melnik_i_matrix_mult_ribbon
