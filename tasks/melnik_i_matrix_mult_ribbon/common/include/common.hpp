#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace melnik_i_matrix_mult_ribbon {

using InType = std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>>;
using OutType = std::vector<std::vector<double>>;
using TestType = std::tuple<int, std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace melnik_i_matrix_mult_ribbon
