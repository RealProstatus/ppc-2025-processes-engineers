#include <gtest/gtest.h>

#include "melnik_i_min_neigh_diff_vec/common/include/common.hpp"
#include "melnik_i_min_neigh_diff_vec/mpi/include/ops_mpi.hpp"
#include "melnik_i_min_neigh_diff_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace melnik_i_min_neigh_diff_vec {

class MelnikIRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MelnikIRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NesterovATestTaskMPI, NesterovATestTaskSEQ>(PPC_SETTINGS_melnik_i_min_neigh_diff_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MelnikIRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MelnikIRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace melnik_i_min_neigh_diff_vec
