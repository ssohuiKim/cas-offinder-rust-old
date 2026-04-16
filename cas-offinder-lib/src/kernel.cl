#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable


//PATTERN_LEN: defined in search.rs
//BLOCKS_PER_EXEC: defined in search.rs
//PATTERN_CHUNK_SIZE: defined in search.rs
//block_ty: defined in search.rs
//MAX_MISMATCHES

#define cdiv(x,y) (((x) + (y) - 1) / (y))
#define uint64_t unsigned long
#define uint32_t unsigned int
#define size_t unsigned long
#define uint16_t unsigned short
#define BLOCKS_AVAIL (sizeof(block_ty) * 2)
#define BLOCKS_PER_PATTERN (cdiv(PATTERN_LEN, BLOCKS_AVAIL))
#define PATTERN_OFFSET (cdiv(PATTERN_LEN, 2*PATTERN_CHUNK_SIZE)*PATTERN_CHUNK_SIZE/sizeof(block_ty))

struct s_match
{
    uint32_t loc;
    uint32_t pattern_idx;
    uint32_t mismatches;
    uint16_t dna_bulge_size;
    uint16_t rna_bulge_size;
};
typedef struct s_match match;

__kernel void find_matches(__global block_ty* genome,
                            __global block_ty* pattern_blocks,
                            uint32_t max_mismatches,
                            __global match* match_buffer,
                           __global int* entrycount)
{
    size_t genome_idx = get_global_id(0) * BLOCKS_PER_EXEC;
    size_t pattern_block_idx = get_global_id(1);
    block_ty shifted_blocks[BLOCKS_PER_PATTERN + BLOCKS_PER_EXEC];
    for (size_t i = 0; i < BLOCKS_PER_PATTERN + BLOCKS_PER_EXEC; i++) {
        shifted_blocks[i] = genome[genome_idx + i];
    }
    // genome is expected to be at least BIGGER than the genome_size
    for (size_t k = 0; k < BLOCKS_AVAIL; k++) {
        uint32_t counts[BLOCKS_PER_EXEC] = {0};
#pragma unroll
        for (size_t l = 0; l < BLOCKS_PER_PATTERN; l++) {
#pragma unroll
            for(size_t o = 0; o < BLOCKS_PER_EXEC; o++){
                counts[o] += popcount(
                  shifted_blocks[l + o] &
                  pattern_blocks[pattern_block_idx * PATTERN_OFFSET + l]
                );
            }
        }
#pragma unroll
        for (size_t l = 0; l < BLOCKS_PER_PATTERN + BLOCKS_PER_EXEC - 1; l++) {
            shifted_blocks[l] =
              (shifted_blocks[l] >> 4) |
              (shifted_blocks[l + 1] << ((BLOCKS_AVAIL - 1) * 4));
        }
        shifted_blocks[BLOCKS_PER_PATTERN + BLOCKS_PER_EXEC - 1] >>= 4;
#pragma unroll
        for(size_t o = 0; o < BLOCKS_PER_EXEC; o++){
            int mismatches = PATTERN_LEN - counts[o];
            if (mismatches <= max_mismatches) {
                int next_idx = atomic_inc(entrycount);
                match next_item = {
                    .loc = (genome_idx + o) * BLOCKS_AVAIL + k,
                    .pattern_idx = pattern_block_idx,
                    .mismatches = mismatches,
                    .dna_bulge_size = 0,
                    .rna_bulge_size = 0,
                };
                match_buffer[next_idx] = next_item;
            }
        }
    }
}
