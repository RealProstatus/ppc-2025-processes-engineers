#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "melnik_i_gauss_block_part/common/include/common.hpp"
#include "melnik_i_gauss_block_part/mpi/include/ops_mpi.hpp"
#include "melnik_i_gauss_block_part/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace melnik_i_gauss_block_part {

namespace {

OutType ApplyGaussianReference(const InType &input) {
  static constexpr std::array<int, 9> kKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  static constexpr int kKernelSum = 16;

  const auto &[data, width, height] = input;
  if (width <= 0 || height <= 0) {
    return {};
  }
  const std::size_t expected = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (data.size() != expected) {
    return {};
  }

  OutType output;
  output.resize(data.size());

  auto idx = [&](int y, int x) {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
  };

  auto clamp = [](int value, int low, int high) { return std::max(low, std::min(value, high)); };

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int accum = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        const int yy = clamp(y + ky, 0, height - 1);
        for (int kx = -1; kx <= 1; ++kx) {
          const int xx = clamp(x + kx, 0, width - 1);
          accum += kKernel[(ky + 1) * 3 + (kx + 1)] * data[idx(yy, xx)];
        }
      }
      output[idx(y, x)] = (accum + kKernelSum / 2) / kKernelSum;
    }
  }

  return output;
}

bool VectorsEqual(const OutType &lhs, const OutType &rhs) {
  return lhs == rhs;
}

}  // namespace

class MelnikIGaussBlockPartFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {  // NOLINT
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto expected = ApplyGaussianReference(input_data_);
    return VectorsEqual(expected, output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

InType MakeImage(int width, int height, const std::vector<int> &data) {
  return {data, width, height};
}

InType MakeRamp(int width, int height, int base = 0) {
  std::vector<int> data(static_cast<std::size_t>(width * height));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      data[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] =
          base + y * 100 + x;
    }
  }
  return {data, width, height};
}

// Predictable, small, exactly divisible cases for running under mpirun -np 1/2/4.
// These do NOT depend on a particular 2D grid choice, they just ensure "nice" splits exist.
const std::array<TestType, 4> kTestParam = {
    std::make_tuple(MakeRamp(4, 4, 0), "divisible_p1_4x4"),
    // For P=2, our grid selector tends to pick 1x2 when H/W≈0.5; 6x3 fits that and divides by 2 along X.
    std::make_tuple(MakeRamp(6, 3, 10), "divisible_p2_6x3"),
    // For P=4, 8x4 divides cleanly for both 1x4 and 2x2 factorizations.
    std::make_tuple(MakeRamp(8, 4, 20), "divisible_p4_8x4"),
    std::make_tuple(MakeImage(3, 3, std::vector<int>{10, 20, 30, 40, 50, 60, 70, 80, 90}), "tiny_3x3"),
};

// Extra rectangular / "not so nice" cases: non-square, non-uniform sizes.
const std::array<TestType, 5> kRectParam = {
    std::make_tuple(MakeRamp(5, 4, 0), "rect_5x4"),  std::make_tuple(MakeRamp(7, 3, 5), "rect_7x3"),
    std::make_tuple(MakeRamp(3, 7, 7), "rect_3x7"),  std::make_tuple(MakeRamp(1, 8, 11), "thin_1x8"),
    std::make_tuple(MakeRamp(8, 1, 13), "thin_8x1"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MelnikIGaussBlockPartMPI, InType>(kTestParam, PPC_SETTINGS_melnik_i_gauss_block_part),
    ppc::util::AddFuncTask<MelnikIGaussBlockPartSEQ, InType>(kTestParam, PPC_SETTINGS_melnik_i_gauss_block_part),
    ppc::util::AddFuncTask<MelnikIGaussBlockPartMPI, InType>(kRectParam, PPC_SETTINGS_melnik_i_gauss_block_part),
    ppc::util::AddFuncTask<MelnikIGaussBlockPartSEQ, InType>(kRectParam, PPC_SETTINGS_melnik_i_gauss_block_part));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kFuncTestName = MelnikIGaussBlockPartFuncTests::PrintFuncTestName<MelnikIGaussBlockPartFuncTests>;

TEST_P(MelnikIGaussBlockPartFuncTests, RunsGaussianBlur) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Gauss3x3Tests, MelnikIGaussBlockPartFuncTests, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace melnik_i_gauss_block_part
