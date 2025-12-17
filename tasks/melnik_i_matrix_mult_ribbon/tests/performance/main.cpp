#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "melnik_i_matrix_mult_ribbon/common/include/common.hpp"
#include "melnik_i_matrix_mult_ribbon/mpi/include/ops_mpi.hpp"
#include "melnik_i_matrix_mult_ribbon/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace melnik_i_matrix_mult_ribbon {

class MelnikIMatrixMultRibbonRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kSize = 800;

 protected:
  void SetUp() override {
    matrix_a_ = std::vector<std::vector<double>>(kSize, std::vector<double>(kSize));
    matrix_b_ = std::vector<std::vector<double>>(kSize, std::vector<double>(kSize));

    for (size_t i = 0; i < kSize; ++i) {
      for (size_t j = 0; j < kSize; ++j) {
        matrix_a_[i][j] = static_cast<double>((i * kSize) + j) * 0.001;
        matrix_b_[i][j] = static_cast<double>(i + j) * 0.002;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty() && !output_data[0].empty();
  }

  InType GetTestInputData() final {
    return std::make_tuple(matrix_a_, matrix_b_);
  }

 private:
  std::vector<std::vector<double>> matrix_a_;
  std::vector<std::vector<double>> matrix_b_;
};

TEST_P(MelnikIMatrixMultRibbonRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, MelnikIMatrixMultRibbonMPI, MelnikIMatrixMultRibbonSEQ>(
    PPC_SETTINGS_melnik_i_matrix_mult_ribbon);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MelnikIMatrixMultRibbonRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MelnikIMatrixMultRibbonRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace melnik_i_matrix_mult_ribbon
