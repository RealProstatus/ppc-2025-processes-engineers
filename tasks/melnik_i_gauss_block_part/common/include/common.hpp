#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace melnik_i_gauss_block_part {

struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;  // 1 - grayscale, 3 - RGB
  std::vector<std::uint8_t> data;

  bool IsShapeValid() const {
    if (width <= 0 || height <= 0) {
      return false;
    }
    if (channels != 1 && channels != 3) {
      return false;
    }
    const std::size_t expected_size =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channels);
    return expected_size == data.size();
  }
};

using InType = Image;
using OutType = Image;
using TestType = std::tuple<Image, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace melnik_i_gauss_block_part

