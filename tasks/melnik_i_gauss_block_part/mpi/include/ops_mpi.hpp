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

  static std::pair<int, int> ComputeProcessGrid(int comm_size, int width, int height);
  static BlockInfo ComputeBlockInfo(int rank, int grid_rows, int grid_cols, int width, int height);
  static BlockInfo ComputeBlockInfoByCoords(int pr, int pc, int grid_rows, int grid_cols, int width, int height);

  static int ClampInt(int v, int lo, int hi);
  static void FillExtendedWithClamp(const std::vector<std::uint8_t> &local, const BlockInfo &blk, int ext_w,
                                    std::vector<std::uint8_t> &ext);

  static void ExchangeHalos(const BlockInfo &blk, int grid_rows, int grid_cols, int rank,
                            const std::vector<BlockInfo> &all_blocks, std::vector<std::uint8_t> &ext);

  static Neighbours ComputeNeighbours(const BlockInfo &blk, int grid_rows, int grid_cols, int rank,
                                      const std::vector<BlockInfo> &all_blocks);
  static void ExchangeRowHalos(const BlockInfo &blk, const Neighbours &nbh, int ext_w, std::vector<std::uint8_t> &ext);
  static void ExchangeColHalos(const BlockInfo &blk, const Neighbours &nbh, int ext_w, std::vector<std::uint8_t> &ext);
  static void ExchangeCornerHalos(const BlockInfo &blk, const Neighbours &nbh, int ext_w,
                                  std::vector<std::uint8_t> &ext);
  static void FixCornersWithoutDiagonal(const BlockInfo &blk, const Neighbours &nbh, int ext_w,
                                        std::vector<std::uint8_t> &ext);

  static void ApplyGaussianFromExtended(const BlockInfo &blk, const std::vector<std::uint8_t> &ext,
                                        std::vector<std::uint8_t> &local_out);
};

}  // namespace melnik_i_gauss_block_part
