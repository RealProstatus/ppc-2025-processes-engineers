#include "melnik_i_gauss_block_part/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "melnik_i_gauss_block_part/common/include/common.hpp"

namespace melnik_i_gauss_block_part {

namespace {

inline std::size_t LocalIndex(int y, int x, int c, int width, int channels) {
  return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) *
             static_cast<std::size_t>(channels) +
         static_cast<std::size_t>(c);
}

}  // namespace

MelnikIGaussBlockPartMPI::MelnikIGaussBlockPartMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool MelnikIGaussBlockPartMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int valid_flag = 0;
  if (rank == 0) {
    valid_flag = GetInput().IsShapeValid() ? 1 : 0;
  }
  MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid_flag == 1;
}

bool MelnikIGaussBlockPartMPI::PreProcessingImpl() {
  return true;
}

std::pair<int, int> MelnikIGaussBlockPartMPI::ComputeProcessGrid(int comm_size, int width, int height) {
  int best_rows = 1;
  int best_cols = comm_size;
  int best_diff = comm_size;

  for (int rows = 1; rows <= comm_size; ++rows) {
    if (comm_size % rows != 0) {
      continue;
    }
    const int cols = comm_size / rows;
    const int diff = std::abs(rows - cols);

    const bool within_height = rows <= height;
    const bool within_width = cols <= width;
    if (within_height && within_width && diff < best_diff) {
      best_rows = rows;
      best_cols = cols;
      best_diff = diff;
    }
  }

  return {best_rows, best_cols};
}

MelnikIGaussBlockPartMPI::BlockInfo MelnikIGaussBlockPartMPI::ComputeBlockInfo(int rank, int grid_rows, int grid_cols,
                                                                               int width, int height) {
  const int row = rank / grid_cols;
  const int col = rank % grid_cols;

  const int base_w = width / grid_cols;
  const int rem_w = width % grid_cols;
  const int base_h = height / grid_rows;
  const int rem_h = height % grid_rows;

  const int local_w = base_w + (col < rem_w ? 1 : 0);
  const int local_h = base_h + (row < rem_h ? 1 : 0);

  const int start_x = col * base_w + std::min(col, rem_w);
  const int start_y = row * base_h + std::min(row, rem_h);

  return BlockInfo{start_x, start_y, local_w, local_h};
}

void MelnikIGaussBlockPartMPI::CopyBlock(const InType &img, const BlockInfo &block, std::vector<std::uint8_t> &buffer) {
  if (block.Empty()) {
    buffer.clear();
    return;
  }

  const int channels = img.channels;
  const std::size_t row_stride_full = static_cast<std::size_t>(img.width) * static_cast<std::size_t>(channels);
  const std::size_t row_stride_block = static_cast<std::size_t>(block.width) * static_cast<std::size_t>(channels);
  buffer.resize(static_cast<std::size_t>(block.height) * row_stride_block);

  for (int y = 0; y < block.height; ++y) {
    const std::size_t src_offset = (static_cast<std::size_t>(block.start_y + y) * row_stride_full) +
                                   static_cast<std::size_t>(block.start_x * channels);
    const std::size_t dst_offset = static_cast<std::size_t>(y) * row_stride_block;
    std::copy(img.data.begin() + src_offset, img.data.begin() + src_offset + row_stride_block,
              buffer.begin() + dst_offset);
  }
}

void MelnikIGaussBlockPartMPI::DistributeImage(const InType &img, const std::vector<BlockInfo> &blocks,
                                               std::vector<std::uint8_t> &local_data, int rank, int comm_size,
                                               int channels) {
  const auto &block = blocks[rank];
  const int local_size = block.Size(channels);
  local_data.resize(static_cast<std::size_t>(std::max(local_size, 0)));

  if (rank == 0) {
    CopyBlock(img, block, local_data);

    std::vector<MPI_Request> requests;
    requests.reserve(static_cast<std::size_t>(comm_size > 0 ? comm_size - 1 : 0));

    for (int dest = 1; dest < comm_size; ++dest) {
      const auto &dst_block = blocks[dest];
      if (dst_block.Empty()) {
        MPI_Request req{};
        MPI_Isend(nullptr, 0, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD, &req);
        requests.push_back(req);
        continue;
      }

      MPI_Datatype block_type{};
      const int sizes[2] = {img.height, img.width * channels};
      const int subsizes[2] = {dst_block.height, dst_block.width * channels};
      const int starts[2] = {dst_block.start_y, dst_block.start_x * channels};
      MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_UINT8_T, &block_type);
      MPI_Type_commit(&block_type);

      MPI_Request req{};
      MPI_Isend(img.data.data(), 1, block_type, dest, 0, MPI_COMM_WORLD, &req);
      MPI_Type_free(&block_type);
      requests.push_back(req);
    }

    if (!requests.empty()) {
      MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
  } else {
    std::uint8_t *recv_ptr = local_size > 0 ? local_data.data() : nullptr;
    MPI_Recv(recv_ptr, std::max(local_size, 0), MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void MelnikIGaussBlockPartMPI::ExchangeHalos(const BlockInfo &block, int grid_rows, int grid_cols, int rank,
                                             int channels, int width, int height,
                                             const std::vector<std::uint8_t> &local_data,
                                             std::vector<std::uint8_t> &extended_block) const {
  (void)grid_rows;  // Grid rows are encoded in rank/grid_cols math
  const int ext_w = block.width + 2;
  const int ext_h = block.height + 2;
  extended_block.assign(static_cast<std::size_t>(ext_w * ext_h * channels), 0);

  if (block.Empty()) {
    return;
  }

  // Copy interior
  for (int y = 0; y < block.height; ++y) {
    for (int x = 0; x < block.width; ++x) {
      for (int c = 0; c < channels; ++c) {
        extended_block[LocalIndex(y + 1, x + 1, c, ext_w, channels)] =
            local_data[LocalIndex(y, x, c, block.width, channels)];
      }
    }
  }

  // Replicate own borders (used when neighbour is absent)
  for (int x = 0; x < block.width; ++x) {
    for (int c = 0; c < channels; ++c) {
      const auto top_val = extended_block[LocalIndex(1, x + 1, c, ext_w, channels)];
      const auto bottom_val = extended_block[LocalIndex(block.height, x + 1, c, ext_w, channels)];
      extended_block[LocalIndex(0, x + 1, c, ext_w, channels)] = top_val;
      extended_block[LocalIndex(block.height + 1, x + 1, c, ext_w, channels)] = bottom_val;
    }
  }

  for (int y = 0; y < block.height; ++y) {
    for (int c = 0; c < channels; ++c) {
      const auto left_val = extended_block[LocalIndex(y + 1, 1, c, ext_w, channels)];
      const auto right_val = extended_block[LocalIndex(y + 1, block.width, c, ext_w, channels)];
      extended_block[LocalIndex(y + 1, 0, c, ext_w, channels)] = left_val;
      extended_block[LocalIndex(y + 1, block.width + 1, c, ext_w, channels)] = right_val;
    }
  }

  for (int c = 0; c < channels; ++c) {
    const auto corner_tl = extended_block[LocalIndex(1, 1, c, ext_w, channels)];
    const auto corner_tr = extended_block[LocalIndex(1, block.width, c, ext_w, channels)];
    const auto corner_bl = extended_block[LocalIndex(block.height, 1, c, ext_w, channels)];
    const auto corner_br = extended_block[LocalIndex(block.height, block.width, c, ext_w, channels)];
    extended_block[LocalIndex(0, 0, c, ext_w, channels)] = corner_tl;
    extended_block[LocalIndex(0, block.width + 1, c, ext_w, channels)] = corner_tr;
    extended_block[LocalIndex(block.height + 1, 0, c, ext_w, channels)] = corner_bl;
    extended_block[LocalIndex(block.height + 1, block.width + 1, c, ext_w, channels)] = corner_br;
  }

  const int up = (block.start_y > 0) ? rank - grid_cols : MPI_PROC_NULL;
  const int down = (block.start_y + block.height < height) ? rank + grid_cols : MPI_PROC_NULL;
  const int left = (block.start_x > 0) ? rank - 1 : MPI_PROC_NULL;
  const int right = (block.start_x + block.width < width) ? rank + 1 : MPI_PROC_NULL;

  const int up_left = (up != MPI_PROC_NULL && left != MPI_PROC_NULL) ? rank - grid_cols - 1 : MPI_PROC_NULL;
  const int up_right = (up != MPI_PROC_NULL && right != MPI_PROC_NULL) ? rank - grid_cols + 1 : MPI_PROC_NULL;
  const int down_left = (down != MPI_PROC_NULL && left != MPI_PROC_NULL) ? rank + grid_cols - 1 : MPI_PROC_NULL;
  const int down_right = (down != MPI_PROC_NULL && right != MPI_PROC_NULL) ? rank + grid_cols + 1 : MPI_PROC_NULL;

  // Exchange rows
  if (block.width > 0 && block.height > 0) {
    const int row_count = block.width * channels;
    const std::uint8_t *top_row = local_data.data();
    const std::uint8_t *bottom_row =
        local_data.data() + (static_cast<std::size_t>(block.height - 1) * block.width * channels);

    MPI_Sendrecv(top_row, row_count, MPI_UINT8_T, up, 0,
                 extended_block.data() + LocalIndex(block.height + 1, 1, 0, ext_w, channels), row_count, MPI_UINT8_T,
                 down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(bottom_row, row_count, MPI_UINT8_T, down, 1,
                 extended_block.data() + LocalIndex(0, 1, 0, ext_w, channels), row_count, MPI_UINT8_T, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Exchange columns
  if (block.height > 0 && block.width > 0) {
    const int col_count = block.height * channels;
    std::vector<std::uint8_t> left_col(static_cast<std::size_t>(col_count));
    std::vector<std::uint8_t> right_col(static_cast<std::size_t>(col_count));

    for (int y = 0; y < block.height; ++y) {
      for (int c = 0; c < channels; ++c) {
        left_col[static_cast<std::size_t>(y * channels + c)] = local_data[LocalIndex(y, 0, c, block.width, channels)];
        right_col[static_cast<std::size_t>(y * channels + c)] =
            local_data[LocalIndex(y, block.width - 1, c, block.width, channels)];
      }
    }

    std::vector<std::uint8_t> recv_left(static_cast<std::size_t>(col_count), 0);
    std::vector<std::uint8_t> recv_right(static_cast<std::size_t>(col_count), 0);

    MPI_Sendrecv(left_col.data(), col_count, MPI_UINT8_T, left, 2, recv_right.data(), col_count, MPI_UINT8_T, right, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(right_col.data(), col_count, MPI_UINT8_T, right, 3, recv_left.data(), col_count, MPI_UINT8_T, left, 3,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (right != MPI_PROC_NULL) {
      for (int y = 0; y < block.height; ++y) {
        for (int c = 0; c < channels; ++c) {
          extended_block[LocalIndex(y + 1, block.width + 1, c, ext_w, channels)] =
              recv_right[static_cast<std::size_t>(y * channels + c)];
        }
      }
    }

    if (left != MPI_PROC_NULL) {
      for (int y = 0; y < block.height; ++y) {
        for (int c = 0; c < channels; ++c) {
          extended_block[LocalIndex(y + 1, 0, c, ext_w, channels)] =
              recv_left[static_cast<std::size_t>(y * channels + c)];
        }
      }
    }
  }

  // Exchange corners
  std::vector<std::uint8_t> corner_buffer(static_cast<std::size_t>(channels), 0);
  std::vector<std::uint8_t> corner_recv(static_cast<std::size_t>(channels), 0);

  auto load_corner = [&](int local_y, int local_x) {
    for (int c = 0; c < channels; ++c) {
      corner_buffer[static_cast<std::size_t>(c)] = local_data[LocalIndex(local_y, local_x, c, block.width, channels)];
    }
  };

  if (block.width > 0 && block.height > 0) {
    load_corner(block.height - 1, block.width - 1);
    MPI_Sendrecv(corner_buffer.data(), channels, MPI_UINT8_T, down_right, 4, corner_recv.data(), channels, MPI_UINT8_T,
                 up_left, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (up_left != MPI_PROC_NULL) {
      for (int c = 0; c < channels; ++c) {
        extended_block[LocalIndex(0, 0, c, ext_w, channels)] = corner_recv[static_cast<std::size_t>(c)];
      }
    }

    load_corner(block.height - 1, 0);
    MPI_Sendrecv(corner_buffer.data(), channels, MPI_UINT8_T, down_left, 5, corner_recv.data(), channels, MPI_UINT8_T,
                 up_right, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (up_right != MPI_PROC_NULL) {
      for (int c = 0; c < channels; ++c) {
        extended_block[LocalIndex(0, block.width + 1, c, ext_w, channels)] = corner_recv[static_cast<std::size_t>(c)];
      }
    }

    load_corner(0, block.width - 1);
    MPI_Sendrecv(corner_buffer.data(), channels, MPI_UINT8_T, up_right, 6, corner_recv.data(), channels, MPI_UINT8_T,
                 down_left, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (down_left != MPI_PROC_NULL) {
      for (int c = 0; c < channels; ++c) {
        extended_block[LocalIndex(block.height + 1, 0, c, ext_w, channels)] = corner_recv[static_cast<std::size_t>(c)];
      }
    }

    load_corner(0, 0);
    MPI_Sendrecv(corner_buffer.data(), channels, MPI_UINT8_T, up_left, 7, corner_recv.data(), channels, MPI_UINT8_T,
                 down_right, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (down_right != MPI_PROC_NULL) {
      for (int c = 0; c < channels; ++c) {
        extended_block[LocalIndex(block.height + 1, block.width + 1, c, ext_w, channels)] =
            corner_recv[static_cast<std::size_t>(c)];
      }
    }
  }
}

void MelnikIGaussBlockPartMPI::ApplyKernelToBlock(const BlockInfo &block, int channels,
                                                  const std::vector<std::uint8_t> &extended_block,
                                                  std::vector<std::uint8_t> &local_output) {
  static constexpr std::array<int, 9> kKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  static constexpr int kKernelSum = 16;

  if (block.Empty()) {
    local_output.clear();
    return;
  }

  const int ext_w = block.width + 2;
  local_output.resize(static_cast<std::size_t>(block.width * block.height * channels));

  for (int y = 0; y < block.height; ++y) {
    for (int x = 0; x < block.width; ++x) {
      for (int c = 0; c < channels; ++c) {
        int accum = 0;
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            const auto value = extended_block[LocalIndex(y + ky, x + kx, c, ext_w, channels)];
            accum += kKernel[ky * 3 + kx] * static_cast<int>(value);
          }
        }
        local_output[LocalIndex(y, x, c, block.width, channels)] =
            static_cast<std::uint8_t>((accum + kKernelSum / 2) / kKernelSum);
      }
    }
  }
}

void MelnikIGaussBlockPartMPI::GatherResult(const std::vector<BlockInfo> &blocks, int channels, int width, int height,
                                            const std::vector<std::uint8_t> &local_output, OutType &out_img, int rank,
                                            int comm_size) const {
  const std::size_t global_size =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channels);
  if (rank == 0) {
    out_img.width = width;
    out_img.height = height;
    out_img.channels = channels;
    out_img.data.assign(global_size, 0);

    const auto &root_block = blocks[0];
    const std::size_t row_stride_block =
        static_cast<std::size_t>(root_block.width) * static_cast<std::size_t>(channels);
    const std::size_t row_stride_full = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
    for (int y = 0; y < root_block.height; ++y) {
      const std::size_t dst_offset = (static_cast<std::size_t>(root_block.start_y + y) * row_stride_full) +
                                     static_cast<std::size_t>(root_block.start_x * channels);
      const std::size_t src_offset = static_cast<std::size_t>(y) * row_stride_block;
      std::copy(local_output.begin() + src_offset, local_output.begin() + src_offset + row_stride_block,
                out_img.data.begin() + dst_offset);
    }

    std::vector<MPI_Request> requests;
    requests.reserve(static_cast<std::size_t>(comm_size > 0 ? comm_size - 1 : 0));
    for (int src = 1; src < comm_size; ++src) {
      const auto &blk = blocks[src];
      if (blk.Empty()) {
        MPI_Request req{};
        MPI_Irecv(nullptr, 0, MPI_UINT8_T, src, 100, MPI_COMM_WORLD, &req);
        requests.push_back(req);
        continue;
      }

      MPI_Datatype block_type{};
      const int sizes[2] = {height, width * channels};
      const int subsizes[2] = {blk.height, blk.width * channels};
      const int starts[2] = {blk.start_y, blk.start_x * channels};
      MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_UINT8_T, &block_type);
      MPI_Type_commit(&block_type);

      MPI_Request req{};
      MPI_Irecv(out_img.data.data(), 1, block_type, src, 100, MPI_COMM_WORLD, &req);
      MPI_Type_free(&block_type);
      requests.push_back(req);
    }

    if (!requests.empty()) {
      MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
  } else {
    const std::uint8_t *send_ptr = local_output.empty() ? nullptr : local_output.data();
    MPI_Send(send_ptr, static_cast<int>(local_output.size()), MPI_UINT8_T, 0, 100, MPI_COMM_WORLD);
  }

  // Broadcast result to every process
  MPI_Bcast(&out_img.width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&out_img.height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&out_img.channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int broadcast_size = static_cast<int>(global_size);
  if (rank != 0) {
    out_img.data.assign(global_size, 0);
  }
  if (broadcast_size > 0) {
    MPI_Bcast(out_img.data.data(), broadcast_size, MPI_UINT8_T, 0, MPI_COMM_WORLD);
  }
}

bool MelnikIGaussBlockPartMPI::RunImpl() {
  int rank = 0;
  int comm_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int width = GetInput().width;
  int height = GetInput().height;
  int channels = GetInput().channels;

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const auto [grid_rows, grid_cols] = ComputeProcessGrid(comm_size, width, height);
  std::vector<BlockInfo> blocks(static_cast<std::size_t>(comm_size));
  for (int r = 0; r < comm_size; ++r) {
    blocks[static_cast<std::size_t>(r)] = ComputeBlockInfo(r, grid_rows, grid_cols, width, height);
  }

  std::vector<std::uint8_t> local_data;
  DistributeImage(GetInput(), blocks, local_data, rank, comm_size, channels);

  std::vector<std::uint8_t> extended_block;
  ExchangeHalos(blocks[static_cast<std::size_t>(rank)], grid_rows, grid_cols, rank, channels, width, height, local_data,
                extended_block);

  std::vector<std::uint8_t> local_output;
  ApplyKernelToBlock(blocks[static_cast<std::size_t>(rank)], channels, extended_block, local_output);

  GatherResult(blocks, channels, width, height, local_output, GetOutput(), rank, comm_size);
  return true;
}

bool MelnikIGaussBlockPartMPI::PostProcessingImpl() {
  return true;
}

}  // namespace melnik_i_gauss_block_part
