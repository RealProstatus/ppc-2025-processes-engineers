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

Image ApplyGaussianReference(const Image &input) {
  static constexpr std::array<int, 9> kKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  static constexpr int kKernelSum = 16;

  Image output;
  output.width = input.width;
  output.height = input.height;
  output.channels = input.channels;
  output.data.resize(input.data.size());

  auto idx = [&](int y, int x, int c) {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(input.width) + static_cast<std::size_t>(x)) *
               static_cast<std::size_t>(input.channels) +
           static_cast<std::size_t>(c);
  };

  auto clamp = [](int value, int low, int high) { return std::max(low, std::min(value, high)); };

  for (int y = 0; y < input.height; ++y) {
    for (int x = 0; x < input.width; ++x) {
      for (int c = 0; c < input.channels; ++c) {
        int accum = 0;
        for (int ky = -1; ky <= 1; ++ky) {
          const int yy = clamp(y + ky, 0, input.height - 1);
          for (int kx = -1; kx <= 1; ++kx) {
            const int xx = clamp(x + kx, 0, input.width - 1);
            accum += kKernel[(ky + 1) * 3 + (kx + 1)] * static_cast<int>(input.data[idx(yy, xx, c)]);
          }
        }
        output.data[idx(y, x, c)] = static_cast<std::uint8_t>((accum + kKernelSum / 2) / kKernelSum);
      }
    }
  }

  return output;
}

bool ImagesEqual(const Image &lhs, const Image &rhs) {
  return lhs.width == rhs.width && lhs.height == rhs.height && lhs.channels == rhs.channels && lhs.data == rhs.data;
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
    return ImagesEqual(expected, output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

Image MakeImage(int width, int height, int channels, const std::vector<std::uint8_t> &data) {
  Image img;
  img.width = width;
  img.height = height;
  img.channels = channels;
  img.data = data;
  return img;
}

Image MakeGradient(int width, int height, int channels) {
  Image img;
  img.width = width;
  img.height = height;
  img.channels = channels;
  img.data.resize(static_cast<std::size_t>(width * height * channels));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        img.data[(static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x)) * channels +
                 static_cast<std::size_t>(c)] = static_cast<std::uint8_t>((x + y + c * 3) % 256);
      }
    }
  }
  return img;
}

Image MakeRandom(int width, int height, int channels, unsigned int seed) {
  Image img;
  img.width = width;
  img.height = height;
  img.channels = channels;
  img.data.resize(static_cast<std::size_t>(width * height * channels));
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto &v : img.data) {
    v = static_cast<std::uint8_t>(dist(gen));
  }
  return img;
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(MakeImage(3, 3, 1, std::vector<std::uint8_t>{10, 20, 30, 40, 50, 60, 70, 80, 90}),
                    "grayscale_small"),
    std::make_tuple(MakeImage(2, 2, 3, std::vector<std::uint8_t>{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}),
                    "rgb_two_by_two"),
    std::make_tuple(MakeGradient(5, 4, 1), "grayscale_gradient"),
    std::make_tuple(MakeGradient(7, 5, 3), "rgb_gradient_non_divisible"),
    std::make_tuple(MakeRandom(8, 3, 3, 1234), "wide_rgb_random"),
    std::make_tuple(MakeRandom(4, 9, 1, 2025), "tall_gray_random"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MelnikIGaussBlockPartMPI, InType>(kTestParam, PPC_SETTINGS_melnik_i_gauss_block_part),
    ppc::util::AddFuncTask<MelnikIGaussBlockPartSEQ, InType>(kTestParam, PPC_SETTINGS_melnik_i_gauss_block_part));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kFuncTestName = MelnikIGaussBlockPartFuncTests::PrintFuncTestName<MelnikIGaussBlockPartFuncTests>;

TEST_P(MelnikIGaussBlockPartFuncTests, RunsGaussianBlur) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Gauss3x3Tests, MelnikIGaussBlockPartFuncTests, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace melnik_i_gauss_block_part
