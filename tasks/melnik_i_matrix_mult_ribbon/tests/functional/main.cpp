#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "melnik_i_matrix_mult_ribbon/common/include/common.hpp"
#include "melnik_i_matrix_mult_ribbon/mpi/include/ops_mpi.hpp"
#include "melnik_i_matrix_mult_ribbon/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace melnik_i_matrix_mult_ribbon {

class MelnikIMatrixMultRibbonRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_matrix_mult";
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    matrix_a_ = std::get<1>(params);
    matrix_b_ = std::get<2>(params);
    expected_ = std::get<3>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_.size()) {
      std::cerr << "\n=== DEBUG OUTPUT ===\n";
      std::cerr << "Size mismatch: output=" << output_data.size() << ", expected=" << expected_.size() << "\n";
      std::cerr << "Output matrix:\n";
      for (size_t i = 0; i < output_data.size(); i++) {
        std::cerr << "  [";
        for (size_t j = 0; j < output_data[i].size(); j++) {
          std::cerr << output_data[i][j];
          if (j < output_data[i].size() - 1) {
            std::cerr << ", ";
          }
        }
        std::cerr << "]\n";
      }
      std::cerr << "Expected matrix:\n";
      for (size_t i = 0; i < expected_.size(); i++) {
        std::cerr << "  [";
        for (size_t j = 0; j < expected_[i].size(); j++) {
          std::cerr << expected_[i][j];
          if (j < expected_[i].size() - 1) {
            std::cerr << ", ";
          }
        }
        std::cerr << "]\n";
      }
      std::cerr << "===================\n";
      return false;
    }
    if (!output_data.empty() && output_data[0].size() != expected_[0].size()) {
      std::cerr << "\n=== DEBUG OUTPUT ===\n";
      std::cerr << "Column size mismatch: output[0]=" << output_data[0].size()
                << ", expected[0]=" << expected_[0].size() << "\n";
      std::cerr << "===================\n";
      return false;
    }

    const double tolerance = 1e-10;
    bool has_mismatch = false;
    size_t mismatch_row = 0;
    size_t mismatch_col = 0;

    for (size_t i = 0; i < expected_.size(); i++) {
      for (size_t j = 0; j < expected_[i].size(); j++) {
        if (std::abs(output_data[i][j] - expected_[i][j]) > tolerance) {
          if (!has_mismatch) {
            mismatch_row = i;
            mismatch_col = j;
            has_mismatch = true;
          }
        }
      }
    }

    if (has_mismatch) {
      std::cerr << "\n=== DEBUG OUTPUT ===\n";
      std::cerr << "Value mismatch at position [" << mismatch_row << "][" << mismatch_col << "]\n";
      std::cerr << "Output value: " << output_data[mismatch_row][mismatch_col] << "\n";
      std::cerr << "Expected value: " << expected_[mismatch_row][mismatch_col] << "\n";
      std::cerr << "Difference: "
                << std::abs(output_data[mismatch_row][mismatch_col] - expected_[mismatch_row][mismatch_col]) << "\n";
      std::cerr << "Tolerance: " << tolerance << "\n";
      std::cerr << "\nFull output matrix:\n";
      for (size_t i = 0; i < output_data.size(); i++) {
        std::cerr << "  [";
        for (size_t j = 0; j < output_data[i].size(); j++) {
          std::cerr << output_data[i][j];
          if (j < output_data[i].size() - 1) {
            std::cerr << ", ";
          }
        }
        std::cerr << "]\n";
      }
      std::cerr << "\nFull expected matrix:\n";
      for (size_t i = 0; i < expected_.size(); i++) {
        std::cerr << "  [";
        for (size_t j = 0; j < expected_[i].size(); j++) {
          std::cerr << expected_[i][j];
          if (j < expected_[i].size() - 1) {
            std::cerr << ", ";
          }
        }
        std::cerr << "]\n";
      }
      std::cerr << "===================\n";
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return std::make_tuple(matrix_a_, matrix_b_);
  }

 private:
  std::vector<std::vector<double>> matrix_a_;
  std::vector<std::vector<double>> matrix_b_;
  std::vector<std::vector<double>> expected_;
};

namespace {

TEST_P(MelnikIMatrixMultRibbonRunFuncTestsProcesses, MatrixMultTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    // 1: Identity * dense = dense
    std::make_tuple(1,
                    std::vector<std::vector<double>>{{1, 0, 0, 0, 0, 0},
                                                     {0, 1, 0, 0, 0, 0},
                                                     {0, 0, 1, 0, 0, 0},
                                                     {0, 0, 0, 1, 0, 0},
                                                     {0, 0, 0, 0, 1, 0},
                                                     {0, 0, 0, 0, 0, 1}},
                    std::vector<std::vector<double>>{{1, 2, 3, 4, 5, 6},
                                                     {7, 8, 9, 10, 11, 12},
                                                     {13, 14, 15, 16, 17, 18},
                                                     {19, 20, 21, 22, 23, 24},
                                                     {25, 26, 27, 28, 29, 30},
                                                     {31, 32, 33, 34, 35, 36}},
                    std::vector<std::vector<double>>{{1, 2, 3, 4, 5, 6},
                                                     {7, 8, 9, 10, 11, 12},
                                                     {13, 14, 15, 16, 17, 18},
                                                     {19, 20, 21, 22, 23, 24},
                                                     {25, 26, 27, 28, 29, 30},
                                                     {31, 32, 33, 34, 35, 36}}),

    // 2: Ones * Ones -> each element 6
    std::make_tuple(2, std::vector<std::vector<double>>(6, std::vector<double>(6, 1.0)),
                    std::vector<std::vector<double>>(6, std::vector<double>(6, 1.0)),
                    std::vector<std::vector<double>>(6, std::vector<double>(6, 6.0))),

    // 3: (2 * I) * (3 * I) = 6 * I
    std::make_tuple(3,
                    std::vector<std::vector<double>>{{2, 0, 0, 0, 0, 0},
                                                     {0, 2, 0, 0, 0, 0},
                                                     {0, 0, 2, 0, 0, 0},
                                                     {0, 0, 0, 2, 0, 0},
                                                     {0, 0, 0, 0, 2, 0},
                                                     {0, 0, 0, 0, 0, 2}},
                    std::vector<std::vector<double>>{{3, 0, 0, 0, 0, 0},
                                                     {0, 3, 0, 0, 0, 0},
                                                     {0, 0, 3, 0, 0, 0},
                                                     {0, 0, 0, 3, 0, 0},
                                                     {0, 0, 0, 0, 3, 0},
                                                     {0, 0, 0, 0, 0, 3}},
                    std::vector<std::vector<double>>{{6, 0, 0, 0, 0, 0},
                                                     {0, 6, 0, 0, 0, 0},
                                                     {0, 0, 6, 0, 0, 0},
                                                     {0, 0, 0, 6, 0, 0},
                                                     {0, 0, 0, 0, 6, 0},
                                                     {0, 0, 0, 0, 0, 6}}),

    // 4: Zero matrix * dense = zero
    std::make_tuple(4, std::vector<std::vector<double>>(6, std::vector<double>(6, 0.0)),
                    std::vector<std::vector<double>>{{1, 2, 3, 4, 5, 6},
                                                     {7, 8, 9, 10, 11, 12},
                                                     {13, 14, 15, 16, 17, 18},
                                                     {19, 20, 21, 22, 23, 24},
                                                     {25, 26, 27, 28, 29, 30},
                                                     {31, 32, 33, 34, 35, 36}},
                    std::vector<std::vector<double>>(6, std::vector<double>(6, 0.0))),

    // 5: A * I = A (incremental rows)
    std::make_tuple(5,
                    std::vector<std::vector<double>>{{1, 2, 3, 4, 5, 6},
                                                     {2, 3, 4, 5, 6, 7},
                                                     {3, 4, 5, 6, 7, 8},
                                                     {4, 5, 6, 7, 8, 9},
                                                     {5, 6, 7, 8, 9, 10},
                                                     {6, 7, 8, 9, 10, 11}},
                    std::vector<std::vector<double>>{{1, 0, 0, 0, 0, 0},
                                                     {0, 1, 0, 0, 0, 0},
                                                     {0, 0, 1, 0, 0, 0},
                                                     {0, 0, 0, 1, 0, 0},
                                                     {0, 0, 0, 0, 1, 0},
                                                     {0, 0, 0, 0, 0, 1}},
                    std::vector<std::vector<double>>{{1, 2, 3, 4, 5, 6},
                                                     {2, 3, 4, 5, 6, 7},
                                                     {3, 4, 5, 6, 7, 8},
                                                     {4, 5, 6, 7, 8, 9},
                                                     {5, 6, 7, 8, 9, 10},
                                                     {6, 7, 8, 9, 10, 11}}),

    // 6: Alternating signs * diagonal scaling
    std::make_tuple(6,
                    std::vector<std::vector<double>>{{1, -1, 1, -1, 1, -1},
                                                     {-1, 1, -1, 1, -1, 1},
                                                     {1, -1, 1, -1, 1, -1},
                                                     {-1, 1, -1, 1, -1, 1},
                                                     {1, -1, 1, -1, 1, -1},
                                                     {-1, 1, -1, 1, -1, 1}},
                    std::vector<std::vector<double>>{{1, 0, 0, 0, 0, 0},
                                                     {0, 2, 0, 0, 0, 0},
                                                     {0, 0, 3, 0, 0, 0},
                                                     {0, 0, 0, 4, 0, 0},
                                                     {0, 0, 0, 0, 5, 0},
                                                     {0, 0, 0, 0, 0, 6}},
                    std::vector<std::vector<double>>{{1, -2, 3, -4, 5, -6},
                                                     {-1, 2, -3, 4, -5, 6},
                                                     {1, -2, 3, -4, 5, -6},
                                                     {-1, 2, -3, 4, -5, 6},
                                                     {1, -2, 3, -4, 5, -6},
                                                     {-1, 2, -3, 4, -5, 6}})};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MelnikIMatrixMultRibbonMPI, InType>(kTestParam, PPC_SETTINGS_melnik_i_matrix_mult_ribbon),
    ppc::util::AddFuncTask<MelnikIMatrixMultRibbonSEQ, InType>(kTestParam, PPC_SETTINGS_melnik_i_matrix_mult_ribbon));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    MelnikIMatrixMultRibbonRunFuncTestsProcesses::PrintFuncTestName<MelnikIMatrixMultRibbonRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMultTests, MelnikIMatrixMultRibbonRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace melnik_i_matrix_mult_ribbon
