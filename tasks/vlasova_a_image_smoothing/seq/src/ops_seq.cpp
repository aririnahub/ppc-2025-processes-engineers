#include "vlasova_a_image_smoothing/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <ranges>

#include "vlasova_a_image_smoothing/common/include/common.hpp"

namespace vlasova_a_image_smoothing {

VlasovaAImageSmoothingSEQ::VlasovaAImageSmoothingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VlasovaAImageSmoothingSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.width <= 0 || input.height <= 0) {
    return false;
  }

  if (input.data.empty()) {
    return false;
  }

  const std::size_t expected_size = static_cast<std::size_t>(input.width) * input.height;
  return input.data.size() == expected_size;
}

bool VlasovaAImageSmoothingSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  width_ = input.width;
  height_ = input.height;
  input_image_ = input.data;
  output_image_.resize(input_image_.size());

  return true;
}

bool VlasovaAImageSmoothingSEQ::RunImpl() {
  const int radius = window_size_ / 2;

  for (int row_idx = 0; row_idx < height_; ++row_idx) {
    for (int col_idx = 0; col_idx < width_; ++col_idx) {
      std::vector<std::uint8_t> neighbors;
      neighbors.reserve(static_cast<std::size_t>(window_size_) * window_size_);

      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          const int neighbor_x = col_idx + dx;
          const int neighbor_y = row_idx + dy;

          if (neighbor_x >= 0 && neighbor_x < width_ && neighbor_y >= 0 && neighbor_y < height_) {
            const std::size_t index = (static_cast<std::size_t>(neighbor_y) * width_) + neighbor_x;
            neighbors.push_back(input_image_[index]);
          }
        }
      }

      if (!neighbors.empty()) {
        std::ranges::sort(neighbors);
        const std::size_t output_index = (static_cast<std::size_t>(row_idx) * width_) + col_idx;
        output_image_[output_index] = neighbors[neighbors.size() / 2];
      } else {
        const std::size_t index = (static_cast<std::size_t>(row_idx) * width_) + col_idx;
        output_image_[index] = input_image_[index];
      }
    }
  }

  return true;
}

bool VlasovaAImageSmoothingSEQ::PostProcessingImpl() {
  GetOutput().width = width_;
  GetOutput().height = height_;
  GetOutput().data = output_image_;

  if (GetOutput().data.empty()) {
    return false;
  }

  const std::size_t expected_size = static_cast<std::size_t>(width_) * height_;
  return GetOutput().data.size() == expected_size;
}

}  // namespace vlasova_a_image_smoothing
