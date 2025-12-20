#include "melnik_i_gauss_block_part/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "melnik_i_gauss_block_part/common/include/common.hpp"

namespace melnik_i_gauss_block_part {

namespace {

inline int Clamp(int value, int low, int high) {
  return std::max(low, std::min(value, high));
}

}  // namespace

MelnikIGaussBlockPartSEQ::MelnikIGaussBlockPartSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool MelnikIGaussBlockPartSEQ::ValidationImpl() {
  return GetInput().IsShapeValid();
}

bool MelnikIGaussBlockPartSEQ::PreProcessingImpl() {
  auto &out = GetOutput();
  const auto &input = GetInput();
  out.width = input.width;
  out.height = input.height;
  out.channels = input.channels;
  out.data.assign(static_cast<std::size_t>(out.width) * static_cast<std::size_t>(out.height) *
                      static_cast<std::size_t>(out.channels),
                  0);
  return true;
}

bool MelnikIGaussBlockPartSEQ::RunImpl() {
  ApplyGaussian(GetInput(), GetOutput());
  return true;
}

bool MelnikIGaussBlockPartSEQ::PostProcessingImpl() { return true; }

void MelnikIGaussBlockPartSEQ::ApplyGaussian(const InType &input, OutType &output) {
  static constexpr std::array<int, 9> kKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  static constexpr int kKernelSum = 16;

  const int width = input.width;
  const int height = input.height;
  const int channels = input.channels;
  output.width = width;
  output.height = height;
  output.channels = channels;
  output.data.resize(input.data.size());

  auto idx = [channels, width](int y, int x, int c) {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) *
           static_cast<std::size_t>(channels) +
           static_cast<std::size_t>(c);
  };

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        int accum = 0;
        for (int ky = -1; ky <= 1; ++ky) {
          const int yy = Clamp(y + ky, 0, height - 1);
          for (int kx = -1; kx <= 1; ++kx) {
            const int xx = Clamp(x + kx, 0, width - 1);
            const int kernel_value = kKernel[(ky + 1) * 3 + (kx + 1)];
            accum += kernel_value * static_cast<int>(input.data[idx(yy, xx, c)]);
          }
        }
        output.data[idx(y, x, c)] = static_cast<std::uint8_t>((accum + kKernelSum / 2) / kKernelSum);
      }
    }
  }
}

}  // namespace melnik_i_gauss_block_part

