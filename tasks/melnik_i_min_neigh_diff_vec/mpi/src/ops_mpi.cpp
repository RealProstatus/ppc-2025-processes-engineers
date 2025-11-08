#include "melnik_i_min_neigh_diff_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <cmath>
#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>

#include "melnik_i_min_neigh_diff_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace melnik_i_min_neigh_diff_vec {

struct LocalResult {
    double min_diff;
    int min_index;
    double boundary_left;
    double boundary_right;
};

void minDiffReduce(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype) {
    (void)datatype; // Явно указываем, что параметр не используется
    LocalResult* in = static_cast<LocalResult*>(invec);
    LocalResult* inout = static_cast<LocalResult*>(inoutvec);
    
    for (int i = 0; i < *len; i++) {
        if (in[i].min_diff < inout[i].min_diff) {
            inout[i].min_diff = in[i].min_diff;
            inout[i].min_index = in[i].min_index;
        }
    }
}

MelnikIMinNeighDiffVecMPI::MelnikIMinNeighDiffVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(-1, -1);
}

bool MelnikIMinNeighDiffVecMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
  if (rank == 0) {
    return GetInput().size() >= 2;
  }

  return true;
}

bool MelnikIMinNeighDiffVecMPI::PreProcessingImpl() {
  return true;
}

bool MelnikIMinNeighDiffVecMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
    
  const std::vector<double>& global_vec = GetInput();
  std::vector<double> local_data;
  int local_size = 0;
  int displacement = 0;
  
  std::vector<int> counts(size);
  std::vector<int> displs(size);
    
  // 1. Распределение данных
  if (rank == 0) {
      int n = global_vec.size();
        
      // Вычисляем размеры блоков для каждого процесса
      int base_size = n / size;
      int remainder = n % size;
        
      for (int i = 0; i < size; i++) {
          counts[i] = base_size + (i < remainder ? 1 : 0);
          displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
      }
        
      // Рассылаем размеры блоков и данные
      for (int i = 1; i < size; i++) {
          MPI_Send(&counts[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
          MPI_Send(&displs[i], 1, MPI_INT, i, 1, MPI_COMM_WORLD);
          MPI_Send(global_vec.data() + displs[i], counts[i], MPI_DOUBLE, i, 2, MPI_COMM_WORLD);
      }
        
      local_size = counts[0];
      displacement = displs[0];
      local_data.assign(global_vec.begin(), global_vec.begin() + local_size);
        
  } else {
      // Получаем размер и смещение от root-процесса
      MPI_Recv(&local_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&displacement, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
      local_data.resize(local_size);
      // Получаем данные
      MPI_Recv(local_data.data(), local_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
    
  // 2. Локальный поиск минимума в сегменте
  LocalResult local_result;
  local_result.min_diff = std::numeric_limits<double>::max();
  local_result.min_index = -1;
    
  if (local_size > 0) {
      local_result.boundary_left = local_data[0];
      local_result.boundary_right = local_data[local_size - 1];
        
      // Поиск минимальной разницы в локальном сегменте
      for (int i = 0; i < local_size - 1; i++) {
          double diff = std::abs(local_data[i + 1] - local_data[i]);
          if (diff < local_result.min_diff) {
              local_result.min_diff = diff;
              local_result.min_index = displacement + i; // Глобальный индекс
          }
      }
  } else {
      local_result.boundary_left = 0;
      local_result.boundary_right = 0;
  }
    
  // 3. Учет граничных пар между процессами
  if (size > 1) {
      // Обмен граничными значениями
      double send_left = local_result.boundary_left;
      double send_right = local_result.boundary_right;
      double recv_left, recv_right;
        
      // Обмен с левым соседом
      if (rank > 0) {
          MPI_Send(&send_left, 1, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD);
          MPI_Recv(&recv_right, 1, MPI_DOUBLE, rank - 1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
          double boundary_diff = std::abs(local_result.boundary_left - recv_right);
          if (boundary_diff < local_result.min_diff) {
              local_result.min_diff = boundary_diff;
              local_result.min_index = displacement - 1; // Индекс последнего элемента левого соседа
          }
      }
        
      // Обмен с правым соседом
      if (rank < size - 1) {
          MPI_Recv(&recv_left, 1, MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Send(&send_right, 1, MPI_DOUBLE, rank + 1, 4, MPI_COMM_WORLD);
            
          double boundary_diff = std::abs(local_result.boundary_right - recv_left);
          if (boundary_diff < local_result.min_diff) {
              local_result.min_diff = boundary_diff;
              local_result.min_index = displacement + local_size - 1; // Индекс последнего элемента текущего процесса
          }
      }
  }
    
  // 4. Глобальная редукция для нахождения общего минимума
  LocalResult global_result;
  global_result.min_diff = std::numeric_limits<double>::max();
  global_result.min_index = -1;
    
  // Используем пользовательскую операцию редукции
  MPI_Datatype local_result_type;
  MPI_Type_contiguous(4, MPI_DOUBLE, &local_result_type);
  MPI_Type_commit(&local_result_type);
    
  MPI_Op min_diff_op;
  MPI_Op_create(minDiffReduce, 1, &min_diff_op);
    
  MPI_Allreduce(&local_result, &global_result, 1, local_result_type, min_diff_op, MPI_COMM_WORLD);
    
  MPI_Op_free(&min_diff_op);
  MPI_Type_free(&local_result_type);
    
  // 5. Установка результатов
  if (global_result.min_index >= 0) {
      GetOutput() = std::make_tuple(global_result.min_index, global_result.min_index + 1);
  } else {
      GetOutput() = std::make_tuple(-1, -1);
  }
    
  MPI_Barrier(MPI_COMM_WORLD);
  return global_result.min_index >= 0;
}

bool MelnikIMinNeighDiffVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace melnik_i_min_neigh_diff_vec