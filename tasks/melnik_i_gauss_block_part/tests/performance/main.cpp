#include <gtest/gtest.h>
#include <stb/stb_image.h>

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

class MelnikIGaussBlockPartPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {  // NOLINT
 public:
  void SetUp() override {
    LoadRealImage();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.width == input_.width && output_data.height == input_.height &&
           output_data.channels == input_.channels && !output_data.data.empty() &&
           output_data.data.size() == input_.data.size();
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  void LoadRealImage() {
    input_path_ = ppc::util::GetAbsoluteTaskPath(PPC_ID_melnik_i_gauss_block_part, "verybig.jpg");

    int width = -1;
    int height = -1;
    int channels = -1;
    unsigned char *raw = stbi_load(input_path_.c_str(), &width, &height, &channels, STBI_rgb);
    if (raw == nullptr) {
      throw std::runtime_error("Failed to load image: " + input_path_ +
                               " reason: " + std::string(stbi_failure_reason()));
    }

    if (width <= 0 || height <= 0) {
      stbi_image_free(raw);
      throw std::runtime_error("Loaded image has non-positive dimensions: " + input_path_);
    }

    input_.width = width;
    input_.height = height;
    input_.channels = STBI_rgb;
    const std::size_t sz = static_cast<std::size_t>(input_.width) * static_cast<std::size_t>(input_.height) *
                           static_cast<std::size_t>(input_.channels);
    if (sz == 0) {
      stbi_image_free(raw);
      throw std::runtime_error("Loaded image has zero size: " + input_path_);
    }
    input_.data.assign(raw, raw + sz);
    stbi_image_free(raw);
  }

  InType input_{};
  std::string input_path_{};
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
