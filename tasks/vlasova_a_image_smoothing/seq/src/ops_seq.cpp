#include "vlasova_a_image_smoothing/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vlasova_a_image_smoothing {

VlasovaAImageSmoothingSEQ::VlasovaAImageSmoothingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  window_size_ = 3;
}

bool VlasovaAImageSmoothingSEQ::ValidationImpl() {
  const auto& input = GetInput();
  
  if (input.width <= 0 || input.height <= 0) {
    return false;
  }
  
  if (input.data.empty()) {
    return false;
  }
  
  size_t expected_size = static_cast<size_t>(input.width) * input.height;
  if (input.data.size() != expected_size) {
    return false;
  }
  
  return true;
}

bool VlasovaAImageSmoothingSEQ::PreProcessingImpl() {
  const auto& input = GetInput();

  width_ = input.width;
  height_ = input.height;
  input_image_ = input.data;
  output_image_.resize(input_image_.size());
  
  return true;
}

bool VlasovaAImageSmoothingSEQ::RunImpl() {
  const int radius = window_size_ / 2;
  
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      std::vector<uint8_t> neighbors;
      neighbors.reserve(window_size_ * window_size_);
      
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          int nx = x + dx;
          int ny = y + dy;
        
          if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
            neighbors.push_back(input_image_[ny * width_ + nx]);
          }
        }
      }
      
      if (!neighbors.empty()) {
        auto mid = neighbors.begin() + neighbors.size() / 2;
        std::nth_element(neighbors.begin(), mid, neighbors.end());
        output_image_[y * width_ + x] = *mid;
      }
      else {
        output_image_[y * width_ + x] = input_image_[y * width_ + x];
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
  
  size_t expected_size = static_cast<size_t>(width_) * height_;
  if (GetOutput().data.size() != expected_size) {
    return false;
  }
  
  return true;
}

}  // namespace vlasova_a_image_smoothing