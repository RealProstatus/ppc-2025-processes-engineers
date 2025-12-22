#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x);
  };

  auto clamp = [](int value, int low, int high) { return std::max(low, std::min(value, high)); };

  for (int wy = 0; wy < height; ++wy) {
    for (int wx = 0; wx < width; ++wx) {
      int accum = 0;
      std::size_t kernel_idx = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        const int yy = clamp(wy + ky, 0, height - 1);
        for (int kx = -1; kx <= 1; ++kx) {
          const int xx = clamp(wx + kx, 0, width - 1);
          accum += kKernel.at(kernel_idx) * data[idx(yy, xx)];
          ++kernel_idx;
        }
      }
      output[idx(wy, wx)] = static_cast<std::uint8_t>((accum + kKernelSum / 2) / kKernelSum);
    }
  }

  return output;
}

bool VectorsEqual(const OutType &lhs, const OutType &rhs) {
  return lhs == rhs;
}

}  // namespace

class MelnikIGaussBlockPartFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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

InType MakeImage(int width, int height, const std::vector<std::uint8_t> &data) {
  return {data, width, height};
}

InType MakeRamp(int width, int height, int base = 0) {
  std::vector<std::uint8_t> data(static_cast<std::size_t>(width * height));
  for (int yy = 0; yy < height; ++yy) {
    for (int xx = 0; xx < width; ++xx) {
      const int vv = base + (yy * 17) + (xx * 3);
      data[(static_cast<std::size_t>(yy) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(xx)] =
          static_cast<std::uint8_t>(vv & 0xFF);
    }
  }
  return {data, width, height};
}

const std::array<TestType, 4> kTestParam = {
    std::make_tuple(MakeRamp(4, 4, 0), "divisible_p1_4x4"),
    std::make_tuple(MakeRamp(6, 3, 10), "divisible_p2_6x3"),
    std::make_tuple(MakeRamp(8, 4, 20), "divisible_p4_8x4"),
    std::make_tuple(MakeImage(3, 3, std::vector<std::uint8_t>{10, 20, 30, 40, 50, 60, 70, 80, 90}), "tiny_3x3"),
};

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
