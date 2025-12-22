#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "melnik_i_gauss_block_part/common/include/common.hpp"
#include "task/include/task.hpp"

namespace melnik_i_gauss_block_part {

class MelnikIGaussBlockPartMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit MelnikIGaussBlockPartMPI(const InType &in);

 private:
  struct BlockInfo {
    int start_x = 0;
    int start_y = 0;
    int width = 0;
    int height = 0;
    [[nodiscard]] bool Empty() const {
      return width <= 0 || height <= 0;
    }
  };

  struct Neighbours {
    int up = 0;
    int down = 0;
    int left = 0;
    int right = 0;
    int up_left = 0;
    int up_right = 0;
    int down_left = 0;
    int down_right = 0;
  };

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Computes optimal 2D process grid (rows x cols) matching image aspect ratio
  static std::pair<int, int> ComputeProcessGrid(int comm_size, int width, int height);
  // Computes block info for a process given its rank
  static BlockInfo ComputeBlockInfo(int rank, int grid_rows, int grid_cols, int width, int height);
  // Computes block info for a process given its grid coordinates (process row, process column)
  static BlockInfo ComputeBlockInfoByCoords(int pr, int pc, int grid_rows, int grid_cols, int width, int height);

  // Clamps value v to range [low, high]
  static int ClampInt(int v, int low, int high);
  // Creates extended buffer (with halo borders) from local block using clamp
  static void FillExtendedWithClamp(const std::vector<std::uint8_t> &local, const BlockInfo &blk, int ext_w,
                                    std::vector<std::uint8_t> &ext);

  // Exchanges halo regions (rows, columns, corners) with neighboring processes
  static void ExchangeHalos(const BlockInfo &blk, int grid_rows, int grid_cols, int rank,
                            const std::vector<BlockInfo> &all_blocks, std::vector<std::uint8_t> &ext);

  // Computes ranks of all 8 neighboring processes (up, down, left, right, and 4 diagonals)
  static Neighbours ComputeNeighbours(const BlockInfo &blk, int grid_rows, int grid_cols, int rank,
                                      const std::vector<BlockInfo> &all_blocks);
  // Exchanges top and bottom row halos with vertical neighbors
  static void ExchangeRowHalos(const BlockInfo &blk, const Neighbours &nbh, int ext_w, std::vector<std::uint8_t> &ext);
  // Exchanges left and right column halos with horizontal neighbors
  static void ExchangeColHalos(const BlockInfo &blk, const Neighbours &nbh, int ext_w, std::vector<std::uint8_t> &ext);
  // Exchanges corner pixel values with diagonal neighbors
  static void ExchangeCornerHalos(const BlockInfo &blk, const Neighbours &nbh, int ext_w,
                                  std::vector<std::uint8_t> &ext);
  // Fixes corner halo values when diagonal neighbor is absent, using adjacent edge halos
  static void FixCornersWithoutDiagonal(const BlockInfo &blk, const Neighbours &nbh, int ext_w,
                                        std::vector<std::uint8_t> &ext);

  // Applies 3x3 Gaussian convolution to extended buffer, producing local output block
  static void ApplyGaussianFromExtended(const BlockInfo &blk, const std::vector<std::uint8_t> &ext,
                                        std::vector<std::uint8_t> &local_out);

  // Broadcasts image dimensions (width, height) from rank 0 to all processes
  static void BroadcastImageSize(int rank, int &width, int &height);
  // Builds block information for all processes in the communicator
  static std::vector<BlockInfo> BuildAllBlocks(int comm_size, int grid_rows, int grid_cols, int width, int height);
  // Sends data blocks from root process to all other processes
  static void SendBlocksToOthers(int comm_size, int width, int height, const std::vector<BlockInfo> &blocks,
                                 const std::vector<std::uint8_t> &root_data);
  // Scatters input image blocks
  static std::vector<std::uint8_t> ScatterBlock(int rank, int comm_size, int width, int height,
                                                const std::vector<BlockInfo> &blocks, const BlockInfo &my_blk,
                                                const std::vector<std::uint8_t> &root_data);
  // Computes local convolution result
  static std::vector<std::uint8_t> ComputeLocal(const BlockInfo &my_blk, int grid_rows, int grid_cols, int rank,
                                                const std::vector<BlockInfo> &blocks,
                                                const std::vector<std::uint8_t> &local_data);
  // Gathers local results from all processes into global output array on rank 0
  static std::vector<std::uint8_t> GatherGlobal(int rank, int comm_size, int width, int height,
                                                const std::vector<BlockInfo> &blocks, const BlockInfo &my_blk,
                                                const std::vector<std::uint8_t> &local_out);
  // Finalizes output: broadcast
  void FinalizeOutput(int rank, int width, int height, std::vector<std::uint8_t> &global_out);
};

}  // namespace melnik_i_gauss_block_part
