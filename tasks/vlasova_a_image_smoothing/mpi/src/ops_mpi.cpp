#include "vlasova_a_image_smoothing/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vlasova_a_image_smoothing {

VlasovaAImageSmoothingMPI::VlasovaAImageSmoothingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  window_size_ = 3;
}

bool VlasovaAImageSmoothingMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
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
  }
  
  int is_valid = 1;
  if (rank == 0) {
    const auto& input = GetInput();
    is_valid = (input.width > 0 && input.height > 0 && !input.data.empty()) ? 1 : 0;
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  return is_valid == 1;
}

bool VlasovaAImageSmoothingMPI::PreProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dimensions[2] = {width_, height_};

  if (rank == 0) {
    const auto& input = GetInput();
    width_ = input.width;
    height_ = input.height;
    dimensions[0] = width_;
    dimensions[1] = height_;
}

  MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&window_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  width_ = dimensions[0];
  height_ = dimensions[1];
  
  return true;
}

uint8_t VlasovaAImageSmoothingMPI::GetPixelMedian(int x, int y, int overlap_start, const std::vector<uint8_t>& local_data) const {
  const int radius = window_size_ / 2;
  std::vector<uint8_t> neighbors;
  neighbors.reserve(window_size_ * window_size_);
  
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      int nx = x + dx;
      int ny = y + dy;
      
      if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
        int local_ny = ny - overlap_start;
        neighbors.push_back(local_data[local_ny * width_ + nx]);
      }
    }
  }
  

  if (!neighbors.empty()) {
    auto mid = neighbors.begin() + neighbors.size() / 2;
    std::nth_element(neighbors.begin(), mid, neighbors.end());
    return *mid;
  }
  else {
    int local_y = y - overlap_start;
    return local_data[local_y * width_ + x];
  }
}

bool VlasovaAImageSmoothingMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  const int radius = window_size_ / 2;
  
  int base_rows = height_ / size;
  int extra_rows = height_ % size;
  
  int start_row = rank * base_rows + std::min(rank, extra_rows);
  int end_row = start_row + base_rows + (rank < extra_rows ? 1 : 0);
  int local_height = end_row - start_row;
  
  if (local_height <= 0) {
    local_height = 0;
    start_row = end_row = 0;
  }

  int overlap_start = std::max(0, start_row - radius);
  int overlap_end = std::min(height_, end_row + radius);
  int overlap_height = overlap_end - overlap_start;
  
  std::vector<uint8_t> full_image;
  if (rank == 0) {
    full_image = GetInput().data;
  }
  
  std::vector<int> sendcounts(size), displs(size);
  if (rank == 0) {
    for (int proc = 0; proc < size; ++proc) {
      int proc_start = proc * base_rows + std::min(proc, extra_rows);
      int proc_end = proc_start + base_rows + (proc < extra_rows ? 1 : 0);
      
      if (proc_end > proc_start) {
        int proc_overlap_start = std::max(0, proc_start - radius);
        int proc_overlap_end = std::min(height_, proc_end + radius);
        sendcounts[proc] = (proc_overlap_end - proc_overlap_start) * width_;
        displs[proc] = proc_overlap_start * width_;
      } else {
        sendcounts[proc] = 0;
        displs[proc] = 0;
      }
    }
  }
  
  std::vector<uint8_t> local_image;
  if (local_height > 0) {
    local_image.resize(static_cast<size_t>(overlap_height) * width_);
  }
  
  MPI_Scatterv((rank == 0 ? full_image.data() : nullptr), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, (local_height > 0 ? local_image.data() : nullptr), overlap_height * width_, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  
  std::vector<uint8_t> local_result;
  if (local_height > 0) {
    local_result.resize(static_cast<size_t>(local_height) * width_);
    
    for (int local_y = 0; local_y < local_height; ++local_y) {
      int global_y = start_row + local_y;
      
      for (int x = 0; x < width_; ++x) {
        local_result[local_y * width_ + x] = GetPixelMedian(x, global_y, overlap_start, local_image);
      }
    }
  }

  std::vector<uint8_t> result_image;
  if (rank == 0) { 
    result_image.resize(static_cast<size_t>(width_) * height_);
  }

  if (rank == 0) {
    for (int proc = 0; proc < size; ++proc) {
      int proc_start = proc * base_rows + std::min(proc, extra_rows);
      int proc_end = proc_start + base_rows + (proc < extra_rows ? 1 : 0);
      sendcounts[proc] = (proc_end - proc_start) * width_;
      displs[proc] = proc_start * width_;
    }
  }
  
  MPI_Gatherv((local_height > 0 ? local_result.data() : nullptr), local_height * width_, MPI_UNSIGNED_CHAR, (rank == 0 ? result_image.data() : nullptr), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  
  if (rank == 0) {
    GetOutput().width = width_;
    GetOutput().height = height_;
    GetOutput().data = result_image;
  } else {
    GetOutput().width = width_;
    GetOutput().height = height_;
  }
  
  int data_size = static_cast<int>(GetOutput().data.size());
  MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().data.resize(data_size);
  }
  
  MPI_Bcast(GetOutput().data.data(), data_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  
  return true;
}

bool VlasovaAImageSmoothingMPI::PostProcessingImpl() {
  const auto& output = GetOutput();
  
  if (output.data.empty()) {
    return false;
  }
  
  size_t expected_size = static_cast<size_t>(width_) * height_;
  if (output.data.size() != expected_size) {
    return false;
  }
  
  return true;
}

}  // namespace vlasova_a_image_smoothing