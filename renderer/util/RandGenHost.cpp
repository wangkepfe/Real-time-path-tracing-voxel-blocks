#include "util/RandGenHost.h"
#include "util/RandGenData.h"
#include "util/DebugUtils.h"

void BlueNoiseRandGeneratorHost::init()
{
    CUDA_CHECK(cudaMalloc((void **)&sobol_256spp_256d, 256 * 256 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void **)&scramblingTile, 128 * 128 * 8 * sizeof(unsigned char)));

#if OPTIMIZED_BLUE_NOISE_SPP != 1
    CUDA_CHECK(cudaMalloc((void **)&rankingTile, 128 * 128 * 8 * sizeof(unsigned char)));
#endif

    CUDA_CHECK(cudaMemcpy(sobol_256spp_256d, h_sobol_256spp_256d, 256 * 256 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scramblingTile, h_scramblingTile, 128 * 128 * 8 * sizeof(unsigned char), cudaMemcpyHostToDevice));

#if OPTIMIZED_BLUE_NOISE_SPP != 1
    CUDA_CHECK(cudaMemcpy(rankingTile, h_rankingTile, 128 * 128 * 8 * sizeof(unsigned char), cudaMemcpyHostToDevice));
#endif
}

void BlueNoiseRandGeneratorHost::clear()
{
    CUDA_CHECK(cudaFree(sobol_256spp_256d));
    CUDA_CHECK(cudaFree(scramblingTile));

#if OPTIMIZED_BLUE_NOISE_SPP != 1
    CUDA_CHECK(cudaFree(rankingTile));
#endif
}