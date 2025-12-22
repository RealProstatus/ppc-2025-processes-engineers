#include "melnik_i_gauss_block_part/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "melnik_i_gauss_block_part/common/include/common.hpp"

namespace melnik_i_gauss_block_part {

namespace {

inline int ClampInt(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

}  // namespace

MelnikIGaussBlockPartSEQ::MelnikIGaussBlockPartSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool MelnikIGaussBlockPartSEQ::ValidationImpl() {
  const auto &[data, width, height] = GetInput();

  const std::size_t expected = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  return data.size() == expected;
}

bool MelnikIGaussBlockPartSEQ::PreProcessingImpl() {
  const auto &[data, width, height] = GetInput();

  (void)data;
  GetOutput().assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
  return true;
}

bool MelnikIGaussBlockPartSEQ::RunImpl() {
  const auto &[data, width, height] = GetInput();

  static constexpr std::array<int, 9> kKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  static constexpr int kSum = 16;

  auto &out = GetOutput();
  out.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int acc = 0;
      int k = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          acc += kKernel[static_cast<std::size_t>(k)] * GetPixelClamped(data, width, height, x + dx, y + dy);
          ++k;
        }
      }
      out[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] =
          (acc + kSum / 2) / kSum;
    }
  }
  return true;
}

bool MelnikIGaussBlockPartSEQ::PostProcessingImpl() {
  return true;
}

int MelnikIGaussBlockPartSEQ::GetPixelClamped(const std::vector<int> &data, int width, int height, int x, int y) {
  const int xx = ClampInt(x, 0, width - 1);
  const int yy = ClampInt(y, 0, height - 1);
  return data[static_cast<std::size_t>(yy) * static_cast<std::size_t>(width) + static_cast<std::size_t>(xx)];
}

}  // namespace melnik_i_gauss_block_part
