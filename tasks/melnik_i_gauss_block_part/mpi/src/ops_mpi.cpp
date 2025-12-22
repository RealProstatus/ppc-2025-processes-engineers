#include "melnik_i_gauss_block_part/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <vector>

#include "melnik_i_gauss_block_part/common/include/common.hpp"
#include "task/include/task.hpp"

namespace melnik_i_gauss_block_part {

namespace {

inline std::size_t Idx(int y, int x, int width) {
  return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x);
}

inline std::size_t ExtIdx(int y, int x, int ext_w) {
  return (static_cast<std::size_t>(y) * static_cast<std::size_t>(ext_w)) + static_cast<std::size_t>(x);
}

}  // namespace

MelnikIGaussBlockPartMPI::Neighbours MelnikIGaussBlockPartMPI::ComputeNeighbours(
    const BlockInfo &blk, int grid_rows, int grid_cols, int rank, const std::vector<BlockInfo> &all_blocks) {
  Neighbours nbh;
  if (blk.Empty()) {
    return nbh;
  }

  const int pr = rank / grid_cols;
  const int pc = rank % grid_cols;

  auto get_rank = [&](int npr, int npc) -> int {
    if (npr < 0 || npr >= grid_rows || npc < 0 || npc >= grid_cols) {
      return MPI_PROC_NULL;
    }
    const int neighbour = (npr * grid_cols) + npc;
    const int blocks_size = static_cast<int>(all_blocks.size());
    if (neighbour < 0 || neighbour >= blocks_size) {
      return MPI_PROC_NULL;
    }
    if (all_blocks[static_cast<std::size_t>(neighbour)].Empty()) {
      return MPI_PROC_NULL;
    }
    return neighbour;
  };

  nbh.up = get_rank(pr - 1, pc);
  nbh.down = get_rank(pr + 1, pc);
  nbh.left = get_rank(pr, pc - 1);
  nbh.right = get_rank(pr, pc + 1);
  nbh.up_left = get_rank(pr - 1, pc - 1);
  nbh.up_right = get_rank(pr - 1, pc + 1);
  nbh.down_left = get_rank(pr + 1, pc - 1);
  nbh.down_right = get_rank(pr + 1, pc + 1);
  return nbh;
}

void MelnikIGaussBlockPartMPI::ExchangeRowHalos(const BlockInfo &blk, const Neighbours &nbh, int ext_w,
                                                std::vector<std::uint8_t> &ext) {
  std::vector<std::uint8_t> recv_row(static_cast<std::size_t>(blk.width), 0);

  MPI_Sendrecv(ext.data() + ExtIdx(1, 1, ext_w), blk.width, MPI_BYTE, nbh.up, 10, recv_row.data(), blk.width, MPI_BYTE,
               nbh.up, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (nbh.up != MPI_PROC_NULL) {
    std::ranges::copy(recv_row, ext.begin() + static_cast<std::ptrdiff_t>(ExtIdx(0, 1, ext_w)));
  }

  MPI_Sendrecv(ext.data() + ExtIdx(blk.height, 1, ext_w), blk.width, MPI_BYTE, nbh.down, 11, recv_row.data(), blk.width,
               MPI_BYTE, nbh.down, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (nbh.down != MPI_PROC_NULL) {
    std::ranges::copy(recv_row, ext.begin() + static_cast<std::ptrdiff_t>(ExtIdx(blk.height + 1, 1, ext_w)));
  }
}

void MelnikIGaussBlockPartMPI::ExchangeColHalos(const BlockInfo &blk, const Neighbours &nbh, int ext_w,
                                                std::vector<std::uint8_t> &ext) {
  std::vector<std::uint8_t> send_col(static_cast<std::size_t>(blk.height), 0);
  std::vector<std::uint8_t> recv_col(static_cast<std::size_t>(blk.height), 0);

  for (int row = 0; row < blk.height; ++row) {
    send_col[static_cast<std::size_t>(row)] = ext[ExtIdx(row + 1, 1, ext_w)];
  }
  MPI_Sendrecv(send_col.data(), blk.height, MPI_BYTE, nbh.left, 20, recv_col.data(), blk.height, MPI_BYTE, nbh.left, 21,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (nbh.left != MPI_PROC_NULL) {
    for (int row = 0; row < blk.height; ++row) {
      ext[ExtIdx(row + 1, 0, ext_w)] = recv_col[static_cast<std::size_t>(row)];
    }
  }

  for (int row = 0; row < blk.height; ++row) {
    send_col[static_cast<std::size_t>(row)] = ext[ExtIdx(row + 1, blk.width, ext_w)];
  }
  MPI_Sendrecv(send_col.data(), blk.height, MPI_BYTE, nbh.right, 21, recv_col.data(), blk.height, MPI_BYTE, nbh.right,
               20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (nbh.right != MPI_PROC_NULL) {
    for (int row = 0; row < blk.height; ++row) {
      ext[ExtIdx(row + 1, blk.width + 1, ext_w)] = recv_col[static_cast<std::size_t>(row)];
    }
  }
}

void MelnikIGaussBlockPartMPI::ExchangeCornerHalos(const BlockInfo &blk, const Neighbours &nbh, int ext_w,
                                                   std::vector<std::uint8_t> &ext) {
  std::uint8_t send_val = 0;
  std::uint8_t recv_val = 0;

  // - up_left <-> down_right : tags (30, 31)
  // - up_right <-> down_left : tags (32, 33)
  send_val = ext[ExtIdx(1, 1, ext_w)];
  MPI_Sendrecv(&send_val, 1, MPI_BYTE, nbh.up_left, 30, &recv_val, 1, MPI_BYTE, nbh.up_left, 31, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  if (nbh.up_left != MPI_PROC_NULL) {
    ext[ExtIdx(0, 0, ext_w)] = recv_val;
  }

  send_val = ext[ExtIdx(1, blk.width, ext_w)];
  MPI_Sendrecv(&send_val, 1, MPI_BYTE, nbh.up_right, 32, &recv_val, 1, MPI_BYTE, nbh.up_right, 33, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  if (nbh.up_right != MPI_PROC_NULL) {
    ext[ExtIdx(0, blk.width + 1, ext_w)] = recv_val;
  }

  send_val = ext[ExtIdx(blk.height, 1, ext_w)];
  MPI_Sendrecv(&send_val, 1, MPI_BYTE, nbh.down_left, 33, &recv_val, 1, MPI_BYTE, nbh.down_left, 32, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  if (nbh.down_left != MPI_PROC_NULL) {
    ext[ExtIdx(blk.height + 1, 0, ext_w)] = recv_val;
  }

  send_val = ext[ExtIdx(blk.height, blk.width, ext_w)];
  MPI_Sendrecv(&send_val, 1, MPI_BYTE, nbh.down_right, 31, &recv_val, 1, MPI_BYTE, nbh.down_right, 30, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  if (nbh.down_right != MPI_PROC_NULL) {
    ext[ExtIdx(blk.height + 1, blk.width + 1, ext_w)] = recv_val;
  }
}

void MelnikIGaussBlockPartMPI::FixCornersWithoutDiagonal(const BlockInfo &blk, const Neighbours &nbh, int ext_w,
                                                         std::vector<std::uint8_t> &ext) {
  const bool has_up = nbh.up != MPI_PROC_NULL;
  const bool has_down = nbh.down != MPI_PROC_NULL;
  const bool has_left = nbh.left != MPI_PROC_NULL;
  const bool has_right = nbh.right != MPI_PROC_NULL;

  if (nbh.up_left == MPI_PROC_NULL) {
    if (has_left) {
      ext[ExtIdx(0, 0, ext_w)] = ext[ExtIdx(1, 0, ext_w)];
    } else if (has_up) {
      ext[ExtIdx(0, 0, ext_w)] = ext[ExtIdx(0, 1, ext_w)];
    } else {
      ext[ExtIdx(0, 0, ext_w)] = ext[ExtIdx(1, 1, ext_w)];
    }
  }

  if (nbh.up_right == MPI_PROC_NULL) {
    if (has_right) {
      ext[ExtIdx(0, blk.width + 1, ext_w)] = ext[ExtIdx(1, blk.width + 1, ext_w)];
    } else if (has_up) {
      ext[ExtIdx(0, blk.width + 1, ext_w)] = ext[ExtIdx(0, blk.width, ext_w)];
    } else {
      ext[ExtIdx(0, blk.width + 1, ext_w)] = ext[ExtIdx(1, blk.width, ext_w)];
    }
  }

  if (nbh.down_left == MPI_PROC_NULL) {
    if (has_left) {
      ext[ExtIdx(blk.height + 1, 0, ext_w)] = ext[ExtIdx(blk.height, 0, ext_w)];
    } else if (has_down) {
      ext[ExtIdx(blk.height + 1, 0, ext_w)] = ext[ExtIdx(blk.height + 1, 1, ext_w)];
    } else {
      ext[ExtIdx(blk.height + 1, 0, ext_w)] = ext[ExtIdx(blk.height, 1, ext_w)];
    }
  }

  if (nbh.down_right == MPI_PROC_NULL) {
    if (has_right) {
      ext[ExtIdx(blk.height + 1, blk.width + 1, ext_w)] = ext[ExtIdx(blk.height, blk.width + 1, ext_w)];
    } else if (has_down) {
      ext[ExtIdx(blk.height + 1, blk.width + 1, ext_w)] = ext[ExtIdx(blk.height + 1, blk.width, ext_w)];
    } else {
      ext[ExtIdx(blk.height + 1, blk.width + 1, ext_w)] = ext[ExtIdx(blk.height, blk.width, ext_w)];
    }
  }
}

MelnikIGaussBlockPartMPI::MelnikIGaussBlockPartMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool MelnikIGaussBlockPartMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int width = 0;
  int height = 0;
  int valid_flag = 0;
  if (rank == 0) {
    const auto &[data, w, h] = GetInput();
    width = w;
    height = h;
    const std::size_t expected =
        (width > 0 && height > 0) ? (static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) : 0U;
    valid_flag = (width > 0 && height > 0 && data.size() == expected) ? 1 : 0;
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid_flag == 1;
}

bool MelnikIGaussBlockPartMPI::PreProcessingImpl() {
  // Enforce "only rank 0 owns full input data": other ranks can drop the buffer to save memory.
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    auto &in = GetInput();
    std::get<0>(in).clear();
    std::get<0>(in).shrink_to_fit();
  }
  return true;
}

int MelnikIGaussBlockPartMPI::ClampInt(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

std::pair<int, int> MelnikIGaussBlockPartMPI::ComputeProcessGrid(int comm_size, int width, int height) {
  // Choose factorization close to square and roughly matching aspect ratio.
  int best_r = 1;
  int best_c = comm_size;
  double best_cost = std::numeric_limits<double>::infinity();

  const double aspect = static_cast<double>(height) / static_cast<double>(width);

  for (int rows = 1; rows <= comm_size; ++rows) {
    if (comm_size % rows != 0) {
      continue;
    }
    const int cols = comm_size / rows;
    const double grid_aspect = static_cast<double>(rows) / static_cast<double>(cols);
    const double cost = std::abs(grid_aspect - aspect) + (0.01 * std::abs(rows - cols));
    if (cost < best_cost) {
      best_cost = cost;
      best_r = rows;
      best_c = cols;
    }
  }

  return {best_r, best_c};
}

MelnikIGaussBlockPartMPI::BlockInfo MelnikIGaussBlockPartMPI::ComputeBlockInfoByCoords(int pr, int pc, int grid_rows,
                                                                                       int grid_cols, int width,
                                                                                       int height) {
  const int base_w = width / grid_cols;
  const int rem_w = width % grid_cols;
  const int base_h = height / grid_rows;
  const int rem_h = height % grid_rows;

  const int local_w = base_w + (pc < rem_w ? 1 : 0);
  const int local_h = base_h + (pr < rem_h ? 1 : 0);

  const int start_x = (pc * base_w) + std::min(pc, rem_w);
  const int start_y = (pr * base_h) + std::min(pr, rem_h);

  return BlockInfo{.start_x = start_x, .start_y = start_y, .width = local_w, .height = local_h};
}

MelnikIGaussBlockPartMPI::BlockInfo MelnikIGaussBlockPartMPI::ComputeBlockInfo(int rank, int grid_rows, int grid_cols,
                                                                               int width, int height) {
  const int pr = rank / grid_cols;
  const int pc = rank % grid_cols;
  return ComputeBlockInfoByCoords(pr, pc, grid_rows, grid_cols, width, height);
}

void MelnikIGaussBlockPartMPI::FillExtendedWithClamp(const std::vector<std::uint8_t> &local, const BlockInfo &blk,
                                                     int ext_w, std::vector<std::uint8_t> &ext) {
  const int ext_h = blk.height + 2;
  ext.assign(static_cast<std::size_t>(ext_w) * static_cast<std::size_t>(ext_h), 0);

  // Interior
  for (int row = 0; row < blk.height; ++row) {
    for (int col = 0; col < blk.width; ++col) {
      ext[ExtIdx(row + 1, col + 1, ext_w)] = local[Idx(row, col, blk.width)];
    }
  }

  // Clamp borders based on own interior
  for (int col = 1; col <= blk.width; ++col) {
    ext[ExtIdx(0, col, ext_w)] = ext[ExtIdx(1, col, ext_w)];
    ext[ExtIdx(blk.height + 1, col, ext_w)] = ext[ExtIdx(blk.height, col, ext_w)];
  }
  for (int row = 1; row <= blk.height; ++row) {
    ext[ExtIdx(row, 0, ext_w)] = ext[ExtIdx(row, 1, ext_w)];
    ext[ExtIdx(row, blk.width + 1, ext_w)] = ext[ExtIdx(row, blk.width, ext_w)];
  }
  ext[ExtIdx(0, 0, ext_w)] = ext[ExtIdx(1, 1, ext_w)];
  ext[ExtIdx(0, blk.width + 1, ext_w)] = ext[ExtIdx(1, blk.width, ext_w)];
  ext[ExtIdx(blk.height + 1, 0, ext_w)] = ext[ExtIdx(blk.height, 1, ext_w)];
  ext[ExtIdx(blk.height + 1, blk.width + 1, ext_w)] = ext[ExtIdx(blk.height, blk.width, ext_w)];
}

void MelnikIGaussBlockPartMPI::ExchangeHalos(const BlockInfo &blk, int grid_rows, int grid_cols, int rank,
                                             const std::vector<BlockInfo> &all_blocks, std::vector<std::uint8_t> &ext) {
  if (blk.Empty()) {
    return;
  }
  const Neighbours nbh = ComputeNeighbours(blk, grid_rows, grid_cols, rank, all_blocks);
  const int ext_w = blk.width + 2;

  ExchangeRowHalos(blk, nbh, ext_w, ext);
  ExchangeColHalos(blk, nbh, ext_w, ext);
  ExchangeCornerHalos(blk, nbh, ext_w, ext);
  FixCornersWithoutDiagonal(blk, nbh, ext_w, ext);
}

void MelnikIGaussBlockPartMPI::ApplyGaussianFromExtended(const BlockInfo &blk, const std::vector<std::uint8_t> &ext,
                                                         std::vector<std::uint8_t> &local_out) {
  static constexpr std::array<int, 9> kKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  static constexpr int kSum = 16;

  if (blk.Empty()) {
    local_out.clear();
    return;
  }

  const int ext_w = blk.width + 2;
  local_out.resize(static_cast<std::size_t>(blk.width) * static_cast<std::size_t>(blk.height));

  for (int row = 0; row < blk.height; ++row) {
    for (int col = 0; col < blk.width; ++col) {
      int acc = 0;
      std::size_t kernel_idx = 0;
      for (int dy = 0; dy < 3; ++dy) {
        for (int dx = 0; dx < 3; ++dx) {
          acc += kKernel.at(kernel_idx) * static_cast<int>(ext[ExtIdx(row + dy, col + dx, ext_w)]);
          ++kernel_idx;
        }
      }
      local_out[Idx(row, col, blk.width)] = static_cast<std::uint8_t>((acc + kSum / 2) / kSum);
    }
  }
}

bool MelnikIGaussBlockPartMPI::RunImpl() {
  int rank = 0;
  int comm_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int width = 0;
  int height = 0;
  if (rank == 0) {
    const auto &[data, w, h] = GetInput();
    (void)data;
    width = w;
    height = h;
  }
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const auto [grid_rows, grid_cols] = ComputeProcessGrid(comm_size, width, height);

  std::vector<BlockInfo> blocks(static_cast<std::size_t>(comm_size));
  for (int rank_idx = 0; rank_idx < comm_size; ++rank_idx) {
    blocks[static_cast<std::size_t>(rank_idx)] = ComputeBlockInfo(rank_idx, grid_rows, grid_cols, width, height);
  }

  const BlockInfo my_blk = blocks[static_cast<std::size_t>(rank)];

  // Scatter blocks: only rank0 reads full data, others receive packed local blocks.
  std::vector<std::uint8_t> local_data;
  if (!my_blk.Empty()) {
    local_data.resize(static_cast<std::size_t>(my_blk.width) * static_cast<std::size_t>(my_blk.height));
  }

  if (rank == 0) {
    const auto &[data, w, h] = GetInput();
    (void)w;
    (void)h;

    // Copy root local block
    if (!my_blk.Empty()) {
      for (int row = 0; row < my_blk.height; ++row) {
        const int global_row = my_blk.start_y + row;
        const std::size_t src_off = Idx(global_row, my_blk.start_x, width);
        const std::size_t dst_off = Idx(row, 0, my_blk.width);
        std::ranges::copy(data.begin() + static_cast<std::ptrdiff_t>(src_off),
                          data.begin() + static_cast<std::ptrdiff_t>(src_off + static_cast<std::size_t>(my_blk.width)),
                          local_data.begin() + static_cast<std::ptrdiff_t>(dst_off));
      }
    }

    for (int dest = 1; dest < comm_size; ++dest) {
      const auto &blk = blocks[static_cast<std::size_t>(dest)];
      if (blk.Empty()) {
        MPI_Send(nullptr, 0, MPI_BYTE, dest, 0, MPI_COMM_WORLD);
        continue;
      }

      MPI_Datatype sub{};
      const std::array<int, 2> sizes = {height, width};
      const std::array<int, 2> subs = {blk.height, blk.width};
      const std::array<int, 2> starts = {blk.start_y, blk.start_x};
      MPI_Type_create_subarray(2, sizes.data(), subs.data(), starts.data(), MPI_ORDER_C, MPI_BYTE, &sub);
      MPI_Type_commit(&sub);

      MPI_Send(data.data(), 1, sub, dest, 0, MPI_COMM_WORLD);
      MPI_Type_free(&sub);
    }
  } else {
    const int recv_count = my_blk.Empty() ? 0 : (my_blk.width * my_blk.height);
    std::uint8_t *recv_ptr = recv_count > 0 ? local_data.data() : nullptr;
    MPI_Recv(recv_ptr, recv_count, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Halo + compute local output
  std::vector<std::uint8_t> local_out;
  if (!my_blk.Empty()) {
    std::vector<std::uint8_t> ext;
    const int ext_w = my_blk.width + 2;
    FillExtendedWithClamp(local_data, my_blk, ext_w, ext);
    ExchangeHalos(my_blk, grid_rows, grid_cols, rank, blocks, ext);
    ApplyGaussianFromExtended(my_blk, ext, local_out);
  }

  // Gather to root
  std::vector<std::uint8_t> global_out;
  if (rank == 0) {
    global_out.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
    // Copy root block
    if (!my_blk.Empty()) {
      for (int row = 0; row < my_blk.height; ++row) {
        const std::size_t dst_off = Idx(my_blk.start_y + row, my_blk.start_x, width);
        const std::size_t src_off = Idx(row, 0, my_blk.width);
        std::ranges::copy(
            local_out.begin() + static_cast<std::ptrdiff_t>(src_off),
            local_out.begin() + static_cast<std::ptrdiff_t>(src_off + static_cast<std::size_t>(my_blk.width)),
            global_out.begin() + static_cast<std::ptrdiff_t>(dst_off));
      }
    }

    for (int src = 1; src < comm_size; ++src) {
      const auto &blk = blocks[static_cast<std::size_t>(src)];
      if (blk.Empty()) {
        MPI_Recv(nullptr, 0, MPI_BYTE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        continue;
      }

      MPI_Datatype sub{};
      const std::array<int, 2> sizes = {height, width};
      const std::array<int, 2> subs = {blk.height, blk.width};
      const std::array<int, 2> starts = {blk.start_y, blk.start_x};
      MPI_Type_create_subarray(2, sizes.data(), subs.data(), starts.data(), MPI_ORDER_C, MPI_BYTE, &sub);
      MPI_Type_commit(&sub);

      MPI_Recv(global_out.data(), 1, sub, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Type_free(&sub);
    }
  } else {
    const int send_count = my_blk.Empty() ? 0 : (my_blk.width * my_blk.height);
    const std::uint8_t *send_ptr = send_count > 0 ? local_out.data() : nullptr;
    MPI_Send(send_ptr, send_count, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
  }

  // In functional tests every rank validates output, so we broadcast.
  // In performance tests broadcasting a full 8K image defeats scalability; keep full output only on rank 0.
  const auto state = GetStateOfTesting();
  const std::size_t total = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (state == ppc::task::StateOfTesting::kFunc) {
    if (rank != 0) {
      global_out.assign(total, 0);
    }
    if (total > 0) {
      MPI_Bcast(global_out.data(), static_cast<int>(total), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    GetOutput() = std::move(global_out);
  } else {
    if (rank == 0) {
      GetOutput() = std::move(global_out);
    } else {
      GetOutput().clear();
    }
  }
  return true;
}

bool MelnikIGaussBlockPartMPI::PostProcessingImpl() {
  return true;
}

}  // namespace melnik_i_gauss_block_part
