#include <cinttypes>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C"
{
	__global__ void increment_if_unique(uint16_t *boards, uint32_t boards_offset, uint32_t num_boards, uint8_t *indices, uint32_t *counter, uint32_t num_counters)
	{
		uint32_t board_idx = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t counter_idx = blockIdx.y * blockDim.y + threadIdx.y;
		if (board_idx >= num_boards || counter_idx >= num_counters)
		{
			return;
		}

		uint16_t *board = &boards[(board_idx + boards_offset) * 81];
		uint8_t *index = &indices[counter_idx * 8];
		uint16_t bits = 0x1ff;
		for (int i = 0; i < 8; i++)
		{
			bits &= board[index[i]];
		}

		if ((bits & (bits - 1)) == 0)
		{
			atomicAdd(counter + counter_idx, 1);
		}
	}

	__global__ void create_histogram(uint16_t *boards, uint32_t num_boards, uint8_t *indices, uint32_t num_indices, uint32_t *histogram)
	{
		uint32_t board_idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (board_idx >= num_boards)
		{
			return;
		}

		uint16_t *board = &boards[board_idx * 81];
		uint32_t values = 0;
		for (int i = 0; i < num_indices; i++)
		{
			values |= board[indices[i]];
		}
		if (__popc(values) != num_indices)
		{
			return;
		}

		for (int cell_index = 0; cell_index < 81; cell_index++)
		{
			uint16_t cell_value = board[cell_index];
			if (cell_value & ~values)
			{
				atomicAdd(histogram + cell_index, 1);
			}
		}
	}
}
