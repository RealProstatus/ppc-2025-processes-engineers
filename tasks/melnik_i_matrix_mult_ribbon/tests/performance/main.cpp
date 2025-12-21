#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

#include "melnik_i_matrix_mult_ribbon/common/include/common.hpp"
#include "melnik_i_matrix_mult_ribbon/mpi/include/ops_mpi.hpp"
#include "melnik_i_matrix_mult_ribbon/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace melnik_i_matrix_mult_ribbon {

class MelnikIMatrixMultRibbonRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kSize = 1024;

 protected:
  void static GenerateMatrix(std::vector<std::vector<double>> &matrix, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);

    for (size_t i = 0; i < kSize; ++i) {
      for (size_t j = 0; j < kSize; ++j) {
        matrix[i][j] = distribution(generator);
      }
    }
  }

  void SetUp() override {
    matrix_a_ = std::vector<std::vector<double>>(kSize, std::vector<double>(kSize));
    matrix_b_ = std::vector<std::vector<double>>(kSize, std::vector<double>(kSize));

    const unsigned int seed_a = 3301;
    const unsigned int seed_b = 3307;

    GenerateMatrix(matrix_a_, seed_a);
    GenerateMatrix(matrix_b_, seed_b);
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
