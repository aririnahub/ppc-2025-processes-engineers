#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace ppc::task {
template <typename InT, typename OutT>
class Task;
}  // namespace ppc::task

namespace vlasova_a_image_smoothing {

struct ImageData {
  std::vector<std::uint8_t> data;
  int width = 0;
  int height = 0;

  std::size_t TotalPixels() const {
    return static_cast<std::size_t>(width) * height;
  }

  bool Empty() const {
    return data.empty();
  }

  bool operator==(const ImageData &other) const {
    return width == other.width && height == other.height && data == other.data;
  }
};

using InType = ImageData;
using OutType = ImageData;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace vlasova_a_image_smoothing
