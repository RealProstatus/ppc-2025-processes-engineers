#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace melnik_i_gauss_block_part {

// Input image is a flat array (row-major): data[y * width + x]
using InType = std::tuple<std::vector<std::uint8_t>, int, int>;  // data, width, height
using OutType = std::vector<std::uint8_t>;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace melnik_i_gauss_block_part
