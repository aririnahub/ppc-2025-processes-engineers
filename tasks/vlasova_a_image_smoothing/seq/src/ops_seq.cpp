#include "vlasova_a_image_smoothing/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <vector>

#include "vlasova_a_image_smoothing/common/include/common.hpp"

namespace vlasova_a_image_smoothing {

// Вспомогательная функция как static
static std::uint8_t CalculatePixelMedian(int col_idx, int row_idx, int width, int height, int window_size,
                                         const std::vector<std::uint8_t> &image) {
  const int radius = window_size / 2;
  std::vector<std::uint8_t> neighbors;
  neighbors.reserve(static_cast<std::size_t>(window_size) * window_size);

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const int neighbor_x = col_idx + dx;
      const int neighbor_y = row_idx + dy;

      if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
        const std::size_t index = (static_cast<std::size_t>(neighbor_y) * width) + neighbor_x;
        neighbors.push_back(image[index]);
      }
    }
  }

  if (!neighbors.empty()) {
    std::ranges::sort(neighbors.begin(), neighbors.end());
    return neighbors[neighbors.size() / 2];
  }

  const std::size_t index = (static_cast<std::size_t>(row_idx) * width) + col_idx;
  return image[index];
}

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
  for (int row_idx = 0; row_idx < height_; ++row_idx) {
    for (int col_idx = 0; col_idx < width_; ++col_idx) {
      const std::size_t output_index = (static_cast<std::size_t>(row_idx) * width_) + col_idx;
      output_image_[output_index] = CalculatePixelMedian(col_idx, row_idx, width_, height_, window_size_, input_image_);
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
