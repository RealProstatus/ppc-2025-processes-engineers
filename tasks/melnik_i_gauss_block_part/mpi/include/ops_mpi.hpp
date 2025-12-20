#pragma once

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

    bool Empty() const {
      return width == 0 || height == 0;
    }
    int Size(int channels) const {
      return width * height * channels;
    }
  };

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::pair<int, int> ComputeProcessGrid(int comm_size, int width, int height);
  static BlockInfo ComputeBlockInfo(int rank, int grid_rows, int grid_cols, int width, int height);
  static void CopyBlock(const InType &img, const BlockInfo &block, std::vector<std::uint8_t> &buffer);

  void DistributeImage(const InType &img, const std::vector<BlockInfo> &blocks, std::vector<std::uint8_t> &local_data,
                       int rank, int comm_size, int channels);
  void ExchangeHalos(const BlockInfo &block, int grid_rows, int grid_cols, int rank, int channels, int width,
                     int height, const std::vector<std::uint8_t> &local_data,
                     std::vector<std::uint8_t> &extended_block) const;
  static void ApplyKernelToBlock(const BlockInfo &block, int channels, const std::vector<std::uint8_t> &extended_block,
                                 std::vector<std::uint8_t> &local_output);
  void GatherResult(const std::vector<BlockInfo> &blocks, int channels, int width, int height,
                    const std::vector<std::uint8_t> &local_output, OutType &out_img, int rank, int comm_size) const;
};

}  // namespace melnik_i_gauss_block_part
