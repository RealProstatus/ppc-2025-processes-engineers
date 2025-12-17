#include "melnik_i_matrix_mult_ribbon/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <ranges>
#include <vector>

#include "melnik_i_matrix_mult_ribbon/common/include/common.hpp"

namespace melnik_i_matrix_mult_ribbon {

MelnikIMatrixMultRibbonMPI::MelnikIMatrixMultRibbonMPI(const InType &in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);

  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = std::vector<std::vector<double>>();

  GetInput() = (proc_rank_ == 0) ? in : InType{};
  if (proc_rank_ == 0) {
    matrix_A_ = std::get<0>(GetInput());
    matrix_B_ = std::get<1>(GetInput());
  }
}

bool MelnikIMatrixMultRibbonMPI::ValidationImpl() {
  bool is_valid = (proc_rank_ == 0) ? ValidateOnRoot() : true;

  int valid_flag = is_valid ? 1 : 0;
  MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (valid_flag == 0) {
    rows_a_ = cols_a_ = rows_b_ = cols_b_ = 0;
    return false;
  }

  std::array<int, 4> dims = {static_cast<int>(rows_a_), static_cast<int>(rows_b_), static_cast<int>(cols_a_),
                             static_cast<int>(cols_b_)};
  MPI_Bcast(dims.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
  rows_a_ = static_cast<std::size_t>(dims[0]);
  rows_b_ = static_cast<std::size_t>(dims[1]);
  cols_a_ = static_cast<std::size_t>(dims[2]);
  cols_b_ = static_cast<std::size_t>(dims[3]);

  return GetOutput().empty();
}

bool MelnikIMatrixMultRibbonMPI::PreProcessingImpl() {
  GetOutput().clear();

  if (proc_rank_ == 0) {
    const auto &[matrix_a, matrix_b] = GetInput();
    rows_a_ = matrix_a.size();
    rows_b_ = matrix_b.size();
    cols_a_ = matrix_a.front().size();
    cols_b_ = matrix_b.front().size();

    flat_a_.clear();
    flat_a_.reserve(rows_a_ * cols_a_);
    for (const auto &row : matrix_a) {
      flat_a_.insert(flat_a_.end(), row.begin(), row.end());
    }

    flat_b_transposed_.assign(cols_b_ * cols_a_, 0.0);
    for (std::size_t row_idx = 0; row_idx < cols_a_; ++row_idx) {
      for (std::size_t col_idx = 0; col_idx < cols_b_; ++col_idx) {
        flat_b_transposed_[(col_idx * cols_a_) + row_idx] = matrix_b[row_idx][col_idx];
      }
    }
  }
  return true;
}

bool MelnikIMatrixMultRibbonMPI::RunImpl() {
  if (rows_a_ == 0 || cols_a_ == 0 || cols_b_ == 0) {
    return false;
  }

  ShareSizes();
  ShareMatrixB();

  std::vector<double> local_a;
  std::vector<int> rows_per_rank;
  const std::size_t local_rows = ScatterRows(local_a, rows_per_rank);

  std::vector<double> local_c(local_rows * cols_b_, 0.0);
  if (!local_a.empty() && local_rows > 0) {
    MultiplyLocal(local_a, local_c);
  }

  GatherAndDistribute(rows_per_rank, local_c);
  return true;
}

bool MelnikIMatrixMultRibbonMPI::PostProcessingImpl() {
  auto &output = GetOutput();
  output.clear();
  output.resize(rows_a_);
  for (std::size_t i = 0; i < rows_a_; ++i) {
    output[i].assign(cols_b_, 0.0);
  }

  const std::size_t total = rows_a_ * cols_b_;
  if (flat_c_.size() >= total) {
    for (std::size_t idx = 0; idx < total; ++idx) {
      const std::size_t r = idx / cols_b_;
      const std::size_t c = idx % cols_b_;
      output[r][c] = flat_c_[idx];
    }
  }

  return true;
}

bool MelnikIMatrixMultRibbonMPI::ValidateOnRoot() {
  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  if (matrix_a.empty() || matrix_b.empty()) {
    return false;
  }

  const std::size_t width_a = matrix_a.front().size();
  const std::size_t width_b = matrix_b.front().size();
  const std::size_t height_b = matrix_b.size();

  if (width_a == 0 || width_b == 0 || width_a != height_b) {
    return false;
  }

  if (!HasUniformRowWidth(matrix_a, width_a)) {
    return false;
  }
  if (!HasUniformRowWidth(matrix_b, width_b)) {
    return false;
  }

  rows_a_ = matrix_a.size();
  cols_a_ = width_a;
  rows_b_ = height_b;
  cols_b_ = width_b;
  return true;
}

bool MelnikIMatrixMultRibbonMPI::HasUniformRowWidth(const std::vector<std::vector<double>> &matrix,
                                                    std::size_t expected_width) {
  return std::ranges::all_of(matrix,
                             [expected_width](const std::vector<double> &row) { return row.size() == expected_width; });
}

void MelnikIMatrixMultRibbonMPI::ShareSizes() {
  std::array<int, 4> sizes = {static_cast<int>(rows_a_), static_cast<int>(rows_b_), static_cast<int>(cols_a_),
                              static_cast<int>(cols_b_)};
  MPI_Bcast(sizes.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
  rows_a_ = static_cast<std::size_t>(sizes[0]);
  rows_b_ = static_cast<std::size_t>(sizes[1]);
  cols_a_ = static_cast<std::size_t>(sizes[2]);
  cols_b_ = static_cast<std::size_t>(sizes[3]);
}

void MelnikIMatrixMultRibbonMPI::ShareMatrixB() {
  const std::size_t total = cols_b_ * cols_a_;
  if (proc_rank_ != 0) {
    flat_b_transposed_.assign(total, 0.0);
  }
  MPI_Bcast(flat_b_transposed_.data(), static_cast<int>(total), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

std::size_t MelnikIMatrixMultRibbonMPI::ScatterRows(std::vector<double> &local_a, std::vector<int> &rows_per_rank) {
  rows_per_rank.resize(proc_num_);
  const int base = static_cast<int>(rows_a_ / static_cast<std::size_t>(proc_num_));
  const int remainder = static_cast<int>(rows_a_ % static_cast<std::size_t>(proc_num_));
  for (int i = 0; i < proc_num_; ++i) {
    rows_per_rank[i] = base + (i < remainder ? 1 : 0);
  }

  std::vector<int> counts(proc_num_);
  std::vector<int> displs(proc_num_);
  for (int i = 0, offset = 0; i < proc_num_; ++i) {
    counts[i] = rows_per_rank[i] * static_cast<int>(cols_a_);
    displs[i] = offset;
    offset += counts[i];
  }

  const int local_count = rows_per_rank[proc_rank_] * static_cast<int>(cols_a_);
  local_a.assign(static_cast<std::size_t>(local_count), 0.0);

  MPI_Scatterv(proc_rank_ == 0 ? flat_a_.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, local_a.data(),
               local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return static_cast<std::size_t>(rows_per_rank[proc_rank_]);
}

void MelnikIMatrixMultRibbonMPI::MultiplyLocal(const std::vector<double> &local_a, std::vector<double> &local_c) const {
  using DiffT = std::ptrdiff_t;
  for (std::size_t row_idx = 0; row_idx < local_c.size() / cols_b_; ++row_idx) {
    const auto a_begin = local_a.begin() + static_cast<DiffT>(row_idx * cols_a_);
    const auto a_end = a_begin + static_cast<DiffT>(cols_a_);
    for (std::size_t col_idx = 0; col_idx < cols_b_; ++col_idx) {
      const auto b_begin = flat_b_transposed_.begin() + static_cast<DiffT>(col_idx * cols_a_);
      local_c[(row_idx * cols_b_) + col_idx] =
          std::transform_reduce(a_begin, a_end, b_begin, 0.0, std::plus<>(), std::multiplies<>());
    }
  }
}

void MelnikIMatrixMultRibbonMPI::GatherAndDistribute(const std::vector<int> &rows_per_rank,
                                                     const std::vector<double> &local_c) {
  std::vector<int> counts(proc_num_);
  std::vector<int> displs(proc_num_);
  for (int i = 0, offset = 0; i < proc_num_; ++i) {
    counts[i] = rows_per_rank[i] * static_cast<int>(cols_b_);
    displs[i] = offset;
    offset += counts[i];
  }

  if (proc_rank_ == 0) {
    flat_c_.assign(rows_a_ * cols_b_, 0.0);
  } else {
    flat_c_.clear();
    flat_c_.shrink_to_fit();
    flat_c_.assign(rows_a_ * cols_b_, 0.0);
  }

  MPI_Gatherv(local_c.data(), counts[proc_rank_], MPI_DOUBLE, proc_rank_ == 0 ? flat_c_.data() : nullptr, counts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Broadcast assembled result so PostProcessing can run uniformly
  MPI_Bcast(flat_c_.data(), static_cast<int>(rows_a_ * cols_b_), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

}  // namespace melnik_i_matrix_mult_ribbon
