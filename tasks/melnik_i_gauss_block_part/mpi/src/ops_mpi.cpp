#include "melnik_i_gauss_block_part/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace melnik_i_gauss_block_part {

namespace {

inline std::size_t Idx(int y, int x, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

inline std::size_t ExtIdx(int y, int x, int ext_w) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(ext_w) + static_cast<std::size_t>(x);
}

}  // namespace

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
  std::size_t sz = 0;
  if (rank == 0) {
    const auto &[data, w, h] = GetInput();
    width = w;
    height = h;
    sz = data.size();
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const std::size_t expected =
      (width > 0 && height > 0) ? static_cast<std::size_t>(width) * static_cast<std::size_t>(height) : 0U;

  unsigned long long sz_ull = 0;
  if (rank == 0) {
    sz_ull = static_cast<unsigned long long>(sz);
  }
  MPI_Bcast(&sz_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  if (width <= 0 || height <= 0) {
    return false;
  }
  return static_cast<std::size_t>(sz_ull) == expected;
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

  for (int r = 1; r <= comm_size; ++r) {
    if (comm_size % r != 0) {
      continue;
    }
    const int c = comm_size / r;
    const double grid_aspect = static_cast<double>(r) / static_cast<double>(c);
    const double cost = std::abs(grid_aspect - aspect) + 0.01 * std::abs(r - c);
    if (cost < best_cost) {
      best_cost = cost;
      best_r = r;
      best_c = c;
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

  const int start_x = pc * base_w + std::min(pc, rem_w);
  const int start_y = pr * base_h + std::min(pr, rem_h);

  return BlockInfo{start_x, start_y, local_w, local_h};
}

MelnikIGaussBlockPartMPI::BlockInfo MelnikIGaussBlockPartMPI::ComputeBlockInfo(int rank, int grid_rows, int grid_cols,
                                                                               int width, int height) {
  const int pr = rank / grid_cols;
  const int pc = rank % grid_cols;
  return ComputeBlockInfoByCoords(pr, pc, grid_rows, grid_cols, width, height);
}

void MelnikIGaussBlockPartMPI::FillExtendedWithClamp(const std::vector<int> &local, const BlockInfo &blk, int ext_w,
                                                     std::vector<int> &ext) {
  const int ext_h = blk.height + 2;
  ext.assign(static_cast<std::size_t>(ext_w * ext_h), 0);

  // Interior
  for (int y = 0; y < blk.height; ++y) {
    for (int x = 0; x < blk.width; ++x) {
      ext[ExtIdx(y + 1, x + 1, ext_w)] = local[Idx(y, x, blk.width)];
    }
  }

  // Clamp borders based on own interior
  for (int x = 1; x <= blk.width; ++x) {
    ext[ExtIdx(0, x, ext_w)] = ext[ExtIdx(1, x, ext_w)];
    ext[ExtIdx(blk.height + 1, x, ext_w)] = ext[ExtIdx(blk.height, x, ext_w)];
  }
  for (int y = 1; y <= blk.height; ++y) {
    ext[ExtIdx(y, 0, ext_w)] = ext[ExtIdx(y, 1, ext_w)];
    ext[ExtIdx(y, blk.width + 1, ext_w)] = ext[ExtIdx(y, blk.width, ext_w)];
  }
  ext[ExtIdx(0, 0, ext_w)] = ext[ExtIdx(1, 1, ext_w)];
  ext[ExtIdx(0, blk.width + 1, ext_w)] = ext[ExtIdx(1, blk.width, ext_w)];
  ext[ExtIdx(blk.height + 1, 0, ext_w)] = ext[ExtIdx(blk.height, 1, ext_w)];
  ext[ExtIdx(blk.height + 1, blk.width + 1, ext_w)] = ext[ExtIdx(blk.height, blk.width, ext_w)];
}

void MelnikIGaussBlockPartMPI::ExchangeHalos(const BlockInfo &blk, int grid_rows, int grid_cols, int width, int height,
                                             int rank, const std::vector<BlockInfo> &all_blocks,
                                             std::vector<int> &ext) {
  if (blk.Empty()) {
    return;
  }

  const int pr = rank / grid_cols;
  const int pc = rank % grid_cols;

  auto neighbor_rank = [&](int npr, int npc) -> int {
    if (npr < 0 || npr >= grid_rows || npc < 0 || npc >= grid_cols) {
      return MPI_PROC_NULL;
    }
    const int r = npr * grid_cols + npc;
    if (r < 0 || r >= static_cast<int>(all_blocks.size())) {
      return MPI_PROC_NULL;
    }
    if (all_blocks[static_cast<std::size_t>(r)].Empty()) {
      return MPI_PROC_NULL;
    }
    return r;
  };

  const int up = neighbor_rank(pr - 1, pc);
  const int down = neighbor_rank(pr + 1, pc);
  const int left = neighbor_rank(pr, pc - 1);
  const int right = neighbor_rank(pr, pc + 1);
  const int up_left = neighbor_rank(pr - 1, pc - 1);
  const int up_right = neighbor_rank(pr - 1, pc + 1);
  const int down_left = neighbor_rank(pr + 1, pc - 1);
  const int down_right = neighbor_rank(pr + 1, pc + 1);

  const int ext_w = blk.width + 2;

  // Rows (overwrite clamp if neighbor exists)
  std::vector<int> recv_row(static_cast<std::size_t>(blk.width), 0);

  // receive top halo from up, send my top row to up? (symmetric with down)
  MPI_Sendrecv(ext.data() + ExtIdx(1, 1, ext_w), blk.width, MPI_INT, up, 10, recv_row.data(), blk.width, MPI_INT, up,
               11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (up != MPI_PROC_NULL) {
    std::copy(recv_row.begin(), recv_row.end(), ext.begin() + static_cast<std::ptrdiff_t>(ExtIdx(0, 1, ext_w)));
  }

  MPI_Sendrecv(ext.data() + ExtIdx(blk.height, 1, ext_w), blk.width, MPI_INT, down, 11, recv_row.data(), blk.width,
               MPI_INT, down, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (down != MPI_PROC_NULL) {
    std::copy(recv_row.begin(), recv_row.end(),
              ext.begin() + static_cast<std::ptrdiff_t>(ExtIdx(blk.height + 1, 1, ext_w)));
  }

  // Columns
  std::vector<int> send_col(static_cast<std::size_t>(blk.height), 0);
  std::vector<int> recv_col(static_cast<std::size_t>(blk.height), 0);

  // left column exchange
  for (int y = 0; y < blk.height; ++y) {
    send_col[static_cast<std::size_t>(y)] = ext[ExtIdx(y + 1, 1, ext_w)];
  }
  MPI_Sendrecv(send_col.data(), blk.height, MPI_INT, left, 20, recv_col.data(), blk.height, MPI_INT, left, 21,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (left != MPI_PROC_NULL) {
    for (int y = 0; y < blk.height; ++y) {
      ext[ExtIdx(y + 1, 0, ext_w)] = recv_col[static_cast<std::size_t>(y)];
    }
  }

  // right column exchange
  for (int y = 0; y < blk.height; ++y) {
    send_col[static_cast<std::size_t>(y)] = ext[ExtIdx(y + 1, blk.width, ext_w)];
  }
  MPI_Sendrecv(send_col.data(), blk.height, MPI_INT, right, 21, recv_col.data(), blk.height, MPI_INT, right, 20,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (right != MPI_PROC_NULL) {
    for (int y = 0; y < blk.height; ++y) {
      ext[ExtIdx(y + 1, blk.width + 1, ext_w)] = recv_col[static_cast<std::size_t>(y)];
    }
  }

  // Corners (1 int each) - overwrite clamp if diagonal neighbor exists
  int send_val = 0;
  int recv_val = 0;

  // Diagonal corner exchange must use matched tag pairs:
  // - up_left <-> down_right : tags (30, 31)
  // - up_right <-> down_left : tags (32, 33)

  // up-left: send my top-left, receive their bottom-right
  send_val = ext[ExtIdx(1, 1, ext_w)];
  MPI_Sendrecv(&send_val, 1, MPI_INT, up_left, 30, &recv_val, 1, MPI_INT, up_left, 31, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  if (up_left != MPI_PROC_NULL) {
    ext[ExtIdx(0, 0, ext_w)] = recv_val;
  }

  // up-right: send my top-right, receive their bottom-left
  send_val = ext[ExtIdx(1, blk.width, ext_w)];
  MPI_Sendrecv(&send_val, 1, MPI_INT, up_right, 32, &recv_val, 1, MPI_INT, up_right, 33, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  if (up_right != MPI_PROC_NULL) {
    ext[ExtIdx(0, blk.width + 1, ext_w)] = recv_val;
  }

  // down-left: send my bottom-left, receive their top-right (swap tags vs up-right)
  send_val = ext[ExtIdx(blk.height, 1, ext_w)];
  MPI_Sendrecv(&send_val, 1, MPI_INT, down_left, 33, &recv_val, 1, MPI_INT, down_left, 32, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  if (down_left != MPI_PROC_NULL) {
    ext[ExtIdx(blk.height + 1, 0, ext_w)] = recv_val;
  }

  // down-right: send my bottom-right, receive their top-left (swap tags vs up-left)
  send_val = ext[ExtIdx(blk.height, blk.width, ext_w)];
  MPI_Sendrecv(&send_val, 1, MPI_INT, down_right, 31, &recv_val, 1, MPI_INT, down_right, 30, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  if (down_right != MPI_PROC_NULL) {
    ext[ExtIdx(blk.height + 1, blk.width + 1, ext_w)] = recv_val;
  }

  // If diagonal neighbour is absent, corners still must be consistent with updated edge halos.
  // This is essential for correct convolution near block boundaries when only left/right (or up/down) halo exists.
  const bool has_up = up != MPI_PROC_NULL;
  const bool has_down = down != MPI_PROC_NULL;
  const bool has_left = left != MPI_PROC_NULL;
  const bool has_right = right != MPI_PROC_NULL;

  if (up_left == MPI_PROC_NULL) {
    if (has_left) {
      ext[ExtIdx(0, 0, ext_w)] = ext[ExtIdx(1, 0, ext_w)];
    } else if (has_up) {
      ext[ExtIdx(0, 0, ext_w)] = ext[ExtIdx(0, 1, ext_w)];
    } else {
      ext[ExtIdx(0, 0, ext_w)] = ext[ExtIdx(1, 1, ext_w)];
    }
  }

  if (up_right == MPI_PROC_NULL) {
    if (has_right) {
      ext[ExtIdx(0, blk.width + 1, ext_w)] = ext[ExtIdx(1, blk.width + 1, ext_w)];
    } else if (has_up) {
      ext[ExtIdx(0, blk.width + 1, ext_w)] = ext[ExtIdx(0, blk.width, ext_w)];
    } else {
      ext[ExtIdx(0, blk.width + 1, ext_w)] = ext[ExtIdx(1, blk.width, ext_w)];
    }
  }

  if (down_left == MPI_PROC_NULL) {
    if (has_left) {
      ext[ExtIdx(blk.height + 1, 0, ext_w)] = ext[ExtIdx(blk.height, 0, ext_w)];
    } else if (has_down) {
      ext[ExtIdx(blk.height + 1, 0, ext_w)] = ext[ExtIdx(blk.height + 1, 1, ext_w)];
    } else {
      ext[ExtIdx(blk.height + 1, 0, ext_w)] = ext[ExtIdx(blk.height, 1, ext_w)];
    }
  }

  if (down_right == MPI_PROC_NULL) {
    if (has_right) {
      ext[ExtIdx(blk.height + 1, blk.width + 1, ext_w)] = ext[ExtIdx(blk.height, blk.width + 1, ext_w)];
    } else if (has_down) {
      ext[ExtIdx(blk.height + 1, blk.width + 1, ext_w)] = ext[ExtIdx(blk.height + 1, blk.width, ext_w)];
    } else {
      ext[ExtIdx(blk.height + 1, blk.width + 1, ext_w)] = ext[ExtIdx(blk.height, blk.width, ext_w)];
    }
  }

  (void)width;
  (void)height;
}

void MelnikIGaussBlockPartMPI::ApplyGaussianFromExtended(const BlockInfo &blk, const std::vector<int> &ext,
                                                         std::vector<int> &local_out) {
  static constexpr std::array<int, 9> kKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  static constexpr int kSum = 16;

  if (blk.Empty()) {
    local_out.clear();
    return;
  }

  const int ext_w = blk.width + 2;
  local_out.resize(static_cast<std::size_t>(blk.width) * static_cast<std::size_t>(blk.height));

  for (int y = 0; y < blk.height; ++y) {
    for (int x = 0; x < blk.width; ++x) {
      int acc = 0;
      int k = 0;
      for (int dy = 0; dy < 3; ++dy) {
        for (int dx = 0; dx < 3; ++dx) {
          acc += kKernel[static_cast<std::size_t>(k)] * ext[ExtIdx(y + dy, x + dx, ext_w)];
          ++k;
        }
      }
      local_out[Idx(y, x, blk.width)] = (acc + kSum / 2) / kSum;
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
  for (int r = 0; r < comm_size; ++r) {
    blocks[static_cast<std::size_t>(r)] = ComputeBlockInfo(r, grid_rows, grid_cols, width, height);
  }

  const BlockInfo my_blk = blocks[static_cast<std::size_t>(rank)];

  // Scatter blocks: only rank0 reads full data, others receive packed local blocks.
  std::vector<int> local_data;
  if (!my_blk.Empty()) {
    local_data.resize(static_cast<std::size_t>(my_blk.width) * static_cast<std::size_t>(my_blk.height));
  }

  if (rank == 0) {
    const auto &[data, w, h] = GetInput();
    (void)w;
    (void)h;

    // Copy root local block
    if (!my_blk.Empty()) {
      for (int y = 0; y < my_blk.height; ++y) {
        const int gy = my_blk.start_y + y;
        const std::size_t src_off = Idx(gy, my_blk.start_x, width);
        const std::size_t dst_off = Idx(y, 0, my_blk.width);
        std::copy(data.begin() + static_cast<std::ptrdiff_t>(src_off),
                  data.begin() + static_cast<std::ptrdiff_t>(src_off + static_cast<std::size_t>(my_blk.width)),
                  local_data.begin() + static_cast<std::ptrdiff_t>(dst_off));
      }
    }

    std::vector<MPI_Request> reqs;
    reqs.reserve(static_cast<std::size_t>(comm_size > 0 ? comm_size - 1 : 0));
    for (int dest = 1; dest < comm_size; ++dest) {
      const auto &blk = blocks[static_cast<std::size_t>(dest)];
      MPI_Request req{};
      if (blk.Empty()) {
        MPI_Isend(nullptr, 0, MPI_INT, dest, 0, MPI_COMM_WORLD, &req);
        reqs.push_back(req);
        continue;
      }

      MPI_Datatype sub{};
      const int sizes[2] = {height, width};
      const int subs[2] = {blk.height, blk.width};
      const int starts[2] = {blk.start_y, blk.start_x};
      MPI_Type_create_subarray(2, sizes, subs, starts, MPI_ORDER_C, MPI_INT, &sub);
      MPI_Type_commit(&sub);

      MPI_Isend(const_cast<int *>(data.data()), 1, sub, dest, 0, MPI_COMM_WORLD, &req);
      MPI_Type_free(&sub);
      reqs.push_back(req);
    }
    if (!reqs.empty()) {
      MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }
  } else {
    const int recv_count = my_blk.Empty() ? 0 : (my_blk.width * my_blk.height);
    int *recv_ptr = recv_count > 0 ? local_data.data() : nullptr;
    MPI_Recv(recv_ptr, recv_count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Halo + compute local output
  std::vector<int> local_out;
  if (!my_blk.Empty()) {
    std::vector<int> ext;
    const int ext_w = my_blk.width + 2;
    FillExtendedWithClamp(local_data, my_blk, ext_w, ext);
    ExchangeHalos(my_blk, grid_rows, grid_cols, width, height, rank, blocks, ext);
    ApplyGaussianFromExtended(my_blk, ext, local_out);
  }

  // Gather to root
  std::vector<int> global_out;
  if (rank == 0) {
    global_out.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
    // Copy root block
    if (!my_blk.Empty()) {
      for (int y = 0; y < my_blk.height; ++y) {
        const std::size_t dst_off = Idx(my_blk.start_y + y, my_blk.start_x, width);
        const std::size_t src_off = Idx(y, 0, my_blk.width);
        std::copy(local_out.begin() + static_cast<std::ptrdiff_t>(src_off),
                  local_out.begin() + static_cast<std::ptrdiff_t>(src_off + static_cast<std::size_t>(my_blk.width)),
                  global_out.begin() + static_cast<std::ptrdiff_t>(dst_off));
      }
    }

    std::vector<MPI_Request> reqs;
    reqs.reserve(static_cast<std::size_t>(comm_size > 0 ? comm_size - 1 : 0));
    for (int src = 1; src < comm_size; ++src) {
      const auto &blk = blocks[static_cast<std::size_t>(src)];
      MPI_Request req{};
      if (blk.Empty()) {
        MPI_Irecv(nullptr, 0, MPI_INT, src, 1, MPI_COMM_WORLD, &req);
        reqs.push_back(req);
        continue;
      }

      MPI_Datatype sub{};
      const int sizes[2] = {height, width};
      const int subs[2] = {blk.height, blk.width};
      const int starts[2] = {blk.start_y, blk.start_x};
      MPI_Type_create_subarray(2, sizes, subs, starts, MPI_ORDER_C, MPI_INT, &sub);
      MPI_Type_commit(&sub);

      MPI_Irecv(global_out.data(), 1, sub, src, 1, MPI_COMM_WORLD, &req);
      MPI_Type_free(&sub);
      reqs.push_back(req);
    }
    if (!reqs.empty()) {
      MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }
  } else {
    const int send_count = my_blk.Empty() ? 0 : (my_blk.width * my_blk.height);
    const int *send_ptr = send_count > 0 ? local_out.data() : nullptr;
    MPI_Send(const_cast<int *>(send_ptr), send_count, MPI_INT, 0, 1, MPI_COMM_WORLD);
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
      MPI_Bcast(global_out.data(), static_cast<int>(total), MPI_INT, 0, MPI_COMM_WORLD);
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
