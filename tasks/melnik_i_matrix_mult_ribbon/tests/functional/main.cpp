#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
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
      return false;
    }
    if (!output_data.empty() && output_data[0].size() != expected_[0].size()) {
      return false;
    }

    const double epsilon = 1e-10;
    for (size_t i = 0; i < expected_.size(); i++) {
      for (size_t j = 0; j < expected_[i].size(); j++) {
        if (std::abs(output_data[i][j] - expected_[i][j]) > epsilon) {
          return false;
        }
      }
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

const std::array<TestType, 11> kTestParam = {
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
                                                     {-1, 2, -3, 4, -5, 6}}),

    // 7: Large magnitude diagonal with negatives
    std::make_tuple(7,
                    std::vector<std::vector<double>>{{1000, 0, 0, 0, 0, 0},
                                                     {0, 2000, 0, 0, 0, 0},
                                                     {0, 0, 3000, 0, 0, 0},
                                                     {0, 0, 0, 4000, 0, 0},
                                                     {0, 0, 0, 0, 5000, 0},
                                                     {0, 0, 0, 0, 0, 6000}},
                    std::vector<std::vector<double>>{{-1, 0, 0, 0, 0, 0},
                                                     {0, -2, 0, 0, 0, 0},
                                                     {0, 0, -3, 0, 0, 0},
                                                     {0, 0, 0, -4, 0, 0},
                                                     {0, 0, 0, 0, -5, 0},
                                                     {0, 0, 0, 0, 0, -6}},
                    std::vector<std::vector<double>>{{-1000, 0, 0, 0, 0, 0},
                                                     {0, -4000, 0, 0, 0, 0},
                                                     {0, 0, -9000, 0, 0, 0},
                                                     {0, 0, 0, -16000, 0, 0},
                                                     {0, 0, 0, 0, -25000, 0},
                                                     {0, 0, 0, 0, 0, -36000}}),

    // 8: Row vector 1x3 times 3x1 -> 1x1
    std::make_tuple(8, std::vector<std::vector<double>>{{1.0, 2.0, 3.0}},
                    std::vector<std::vector<double>>{{4.0}, {5.0}, {6.0}}, std::vector<std::vector<double>>{{32.0}}),

    // 9: Single element 1x1 multiplication
    std::make_tuple(9, std::vector<std::vector<double>>{{7.0}}, std::vector<std::vector<double>>{{3.0}},
                    std::vector<std::vector<double>>{{21.0}})};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MelnikIMatrixMultRibbonMPI, InType>(kTestParam, PPC_SETTINGS_melnik_i_matrix_mult_ribbon),
    ppc::util::AddFuncTask<MelnikIMatrixMultRibbonSEQ, InType>(kTestParam, PPC_SETTINGS_melnik_i_matrix_mult_ribbon));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    MelnikIMatrixMultRibbonRunFuncTestsProcesses::PrintFuncTestName<MelnikIMatrixMultRibbonRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMultTests, MelnikIMatrixMultRibbonRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace melnik_i_matrix_mult_ribbon
