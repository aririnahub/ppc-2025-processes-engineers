#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace vlasova_a_image_smoothing {

struct ImageData {
  std::vector<uint8_t> data;
  int width = 0;
  int height = 0;
  
  size_t total_pixels() const { return static_cast<size_t>(width) * height; }
  bool empty() const { return data.empty(); }
  
  bool operator==(const ImageData& other) const {
    return width == other.width && 
           height == other.height && 
           data == other.data;
  }
};

using InType = ImageData;
using OutType = ImageData;
using TestType = std::tuple<int, std::string>; 
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace vlasova_a_image_smoothing