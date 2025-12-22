#pragma once

#include <cstdint>
#include <vector>

#include "task/include/task.hpp"
#include "vlasova_a_image_smoothing/common/include/common.hpp"

namespace vlasova_a_image_smoothing {

class VlasovaAImageSmoothingMPI : public ppc::task::Task<InType, OutType> {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VlasovaAImageSmoothingMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::uint8_t GetPixelMedian(int col_idx, int row_idx, int overlap_start,
                              const std::vector<std::uint8_t> &local_data) const;

  int width_ = 0;
  int height_ = 0;
  int window_size_ = 3;
};

}  // namespace vlasova_a_image_smoothing
