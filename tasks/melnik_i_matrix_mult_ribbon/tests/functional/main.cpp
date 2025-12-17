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

const std::array<TestType, 25> kTestParam = {
    std::make_tuple(1, std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{5, 6}, {7, 8}},
                    std::vector<std::vector<double>>{{19, 22}, {43, 50}}),

    std::make_tuple(2, std::vector<std::vector<double>>{{1, 0}, {0, 1}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}}, std::vector<std::vector<double>>{{1, 2}, {3, 4}}),

    std::make_tuple(3, std::vector<std::vector<double>>{{1, 1}, {1, 1}},
                    std::vector<std::vector<double>>{{1, 1}, {1, 1}}, std::vector<std::vector<double>>{{2, 2}, {2, 2}}),

    std::make_tuple(4, std::vector<std::vector<double>>{{2, 0}, {0, 2}},
                    std::vector<std::vector<double>>{{3, 0}, {0, 3}}, std::vector<std::vector<double>>{{6, 0}, {0, 6}}),

    std::make_tuple(5, std::vector<std::vector<double>>{{1, 2, 3}}, std::vector<std::vector<double>>{{4}, {5}, {6}},
                    std::vector<std::vector<double>>{{32}}),

    std::make_tuple(6, std::vector<std::vector<double>>{{1}, {2}, {3}}, std::vector<std::vector<double>>{{4, 5, 6}},
                    std::vector<std::vector<double>>{{4, 5, 6}, {8, 10, 12}, {12, 15, 18}}),

    std::make_tuple(7, std::vector<std::vector<double>>{{1}}, std::vector<std::vector<double>>{{1}},
                    std::vector<std::vector<double>>{{1}}),

    std::make_tuple(8, std::vector<std::vector<double>>{{0, 0}, {0, 0}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}}, std::vector<std::vector<double>>{{0, 0}, {0, 0}}),

    std::make_tuple(9, std::vector<std::vector<double>>{{2}}, std::vector<std::vector<double>>{{3}},
                    std::vector<std::vector<double>>{{6}}),

    std::make_tuple(10, std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}},
                    std::vector<std::vector<double>>{{7, 8}, {9, 10}, {11, 12}},
                    std::vector<std::vector<double>>{{58, 64}, {139, 154}}),

    std::make_tuple(11, std::vector<std::vector<double>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                    std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                    std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}),

    std::make_tuple(12, std::vector<std::vector<double>>{{0.5, 0.5}, {0.5, 0.5}},
                    std::vector<std::vector<double>>{{2, 4}, {6, 8}}, std::vector<std::vector<double>>{{4, 6}, {4, 6}}),

    std::make_tuple(13, std::vector<std::vector<double>>{{1, 1}, {1, 1}, {1, 1}},
                    std::vector<std::vector<double>>{{1, 1, 1}, {1, 1, 1}},
                    std::vector<std::vector<double>>{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}}),

    std::make_tuple(14, std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}},
                    std::vector<std::vector<double>>{{7, 8, 9}, {10, 11, 12}},
                    std::vector<std::vector<double>>{{27, 30, 33}, {61, 68, 75}, {95, 106, 117}}),

    std::make_tuple(15, std::vector<std::vector<double>>{{0.1, 0.2}, {0.3, 0.4}},
                    std::vector<std::vector<double>>{{5, 6}, {7, 8}},
                    std::vector<std::vector<double>>{{1.9, 2.2}, {4.3, 5.0}}),

    std::make_tuple(16, std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{0, 0}, {0, 0}}, std::vector<std::vector<double>>{{0, 0}, {0, 0}}),

    std::make_tuple(17, std::vector<std::vector<double>>{{1, 0}, {0, 0}},
                    std::vector<std::vector<double>>{{0, 1}, {0, 0}}, std::vector<std::vector<double>>{{0, 1}, {0, 0}}),

    std::make_tuple(18, std::vector<std::vector<double>>{{-1, 2}, {3, -4}},
                    std::vector<std::vector<double>>{{5, -6}, {-7, 8}},
                    std::vector<std::vector<double>>{{-19, 22}, {43, -50}}),

    std::make_tuple(19, std::vector<std::vector<double>>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}},
                    std::vector<std::vector<double>>{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}},
                    std::vector<std::vector<double>>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}),

    std::make_tuple(20, std::vector<std::vector<double>>{{1, 2, 3, 4}},
                    std::vector<std::vector<double>>{{1}, {2}, {3}, {4}}, std::vector<std::vector<double>>{{30}}),

    std::make_tuple(21, std::vector<std::vector<double>>{{1}, {2}, {3}, {4}},
                    std::vector<std::vector<double>>{{1, 2, 3, 4}},
                    std::vector<std::vector<double>>{{1, 2, 3, 4}, {2, 4, 6, 8}, {3, 6, 9, 12}, {4, 8, 12, 16}}),

    std::make_tuple(22, std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}},
                    std::vector<std::vector<double>>{{1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}},
                    std::vector<std::vector<double>>{
                        {1, 2, 3, 0, 0}, {4, 5, 6, 0, 0}, {7, 8, 9, 0, 0}, {10, 11, 12, 0, 0}, {13, 14, 15, 0, 0}}),

    std::make_tuple(23, std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}, {7, 8}},
                    std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}},
                    std::vector<std::vector<double>>{{9, 12, 15}, {19, 26, 33}, {29, 40, 51}, {39, 54, 69}}),

    std::make_tuple(24, std::vector<std::vector<double>>{{2, 2}, {2, 2}},
                    std::vector<std::vector<double>>{{3, 3}, {3, 3}},
                    std::vector<std::vector<double>>{{12, 12}, {12, 12}}),

    std::make_tuple(25, std::vector<std::vector<double>>{{0.001, 0.002}, {0.003, 0.004}},
                    std::vector<std::vector<double>>{{1000, 2000}, {3000, 4000}},
                    std::vector<std::vector<double>>{{7, 10}, {15, 22}})};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MelnikIMatrixMultRibbonMPI, InType>(kTestParam, PPC_SETTINGS_melnik_i_matrix_mult_ribbon),
    ppc::util::AddFuncTask<MelnikIMatrixMultRibbonSEQ, InType>(kTestParam, PPC_SETTINGS_melnik_i_matrix_mult_ribbon));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    MelnikIMatrixMultRibbonRunFuncTestsProcesses::PrintFuncTestName<MelnikIMatrixMultRibbonRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMultTests, MelnikIMatrixMultRibbonRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace melnik_i_matrix_mult_ribbon
