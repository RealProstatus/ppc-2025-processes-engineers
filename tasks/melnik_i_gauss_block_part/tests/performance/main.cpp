#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "melnik_i_gauss_block_part/common/include/common.hpp"
#include "melnik_i_gauss_block_part/mpi/include/ops_mpi.hpp"
#include "melnik_i_gauss_block_part/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace melnik_i_gauss_block_part {

class MelnikIGaussBlockPartPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    LoadRealImage();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const int rank = ppc::util::GetMPIRank();
    if (rank != 0) {
      // In perf mode MPI implementation may keep output only on rank 0 to avoid huge broadcasts
      return true;
    }
    const auto &[data, width, height] = input_;
    (void)width;
    (void)height;
    return !output_data.empty() && output_data.size() == data.size();
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  void LoadRealImage() {
    const int rank = ppc::util::GetMPIRank();
    if (rank != 0) {
      // Only rank 0 owns full input data
      input_ = {std::vector<std::uint8_t>{}, 0, 0};
      return;
    }
    input_path_ = ppc::util::GetAbsoluteTaskPath(PPC_ID_melnik_i_gauss_block_part, "verybig.jpg");

    int width = -1;
    int height = -1;
    int channels = -1;
    unsigned char *raw = stbi_load(input_path_.c_str(), &width, &height, &channels, STBI_grey);
    if (raw == nullptr) {
      throw std::runtime_error("Failed to load image: " + input_path_ +
                               " reason: " + std::string(stbi_failure_reason()));
    }

    if (width <= 0 || height <= 0) {
      stbi_image_free(raw);
      throw std::runtime_error("Loaded image has non-positive dimensions: " + input_path_);
    }

    const std::size_t sz = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (sz == 0) {
      stbi_image_free(raw);
      throw std::runtime_error("Loaded image has zero size: " + input_path_);
    }

    std::vector<std::uint8_t> data(sz);
    std::copy(raw, raw + sz, data.begin());
    stbi_image_free(raw);

    input_ = {data, width, height};
  }

  InType input_;
  std::string input_path_;
};

TEST_P(MelnikIGaussBlockPartPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, MelnikIGaussBlockPartMPI, MelnikIGaussBlockPartSEQ>(
    PPC_SETTINGS_melnik_i_gauss_block_part);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = MelnikIGaussBlockPartPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MelnikIGaussBlockPartPerfTests, kGtestValues, kPerfTestName);

}  // namespace melnik_i_gauss_block_part
