#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "vehicle/DriftItem.hpp"

inline void check(cudaError_t error) {
    if(error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

#include "compaction_kernels.cuh"

int main(int argc, char** argv) {
    if(argc != 4) return 2;
    const std::size_t count = std::stoull(argv[1]);
    const unsigned dead_percent = std::stoul(argv[2]);
    const int repeats = std::stoi(argv[3]);
    if(count == 0 || count > 10000000 || dead_percent > 100 || repeats < 1) return 2;
    const std::size_t threads = 256;
    const std::size_t blocks = (count + threads - 1) / threads;
    std::size_t prefix_threads = 1;
    while(prefix_threads < blocks) prefix_threads *= 2;
    if(prefix_threads > 1024) return 2;

    std::vector<PredictionParticle> input;
    std::vector<unsigned char> flags(count);
    std::vector<std::size_t> expected;
    input.reserve(count);
    for(std::size_t i = 0; i < count; ++i) {
        input.emplace_back(i, float(i % 1000), float(i / 1000), 1.f);
        flags[i] = ((i * 37u) % 100) < dead_percent;
        if(!flags[i]) expected.push_back(i);
    }

    PredictionParticle *particles = nullptr, *buffer = nullptr;
    unsigned char* remove = nullptr;
    unsigned int *totals = nullptr, *prefix = nullptr, *remainder = nullptr;
    check(cudaMalloc(&particles, count * sizeof(PredictionParticle)));
    check(cudaMalloc(&buffer, count * sizeof(PredictionParticle)));
    check(cudaMalloc(&remove, count));
    check(cudaMalloc(&totals, prefix_threads * sizeof(unsigned int)));
    check(cudaMalloc(&prefix, blocks * threads * sizeof(unsigned int)));
    check(cudaMalloc(&remainder, sizeof(unsigned int)));
    check(cudaMemcpy(particles, input.data(), count * sizeof(PredictionParticle), cudaMemcpyHostToDevice));
    check(cudaMemcpy(remove, flags.data(), count, cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    std::vector<int> modes{0, 1, 2};
    for(int trial = -1; trial < repeats; ++trial) {
        std::mt19937 rng(17 + trial);
        std::shuffle(modes.begin(), modes.end(), rng);
        for(int mode : modes) {
            const auto wall_start = std::chrono::steady_clock::now();
            check(cudaEventRecord(start));
            check(cudaMemset(remainder, 0, sizeof(unsigned int)));
            if(mode == 0) {
                particle_cull_atomic_kernel<<<blocks, threads>>>(particles, buffer, remainder, remove, count);
            } else {
                check(cudaMemset(totals, 0, prefix_threads * sizeof(unsigned int)));
                check(cudaMemset(prefix, 0, blocks * threads * sizeof(unsigned int)));
                if(mode == 1) {
                    p_cull_prefixlocal_kernel<<<blocks, threads>>>(remove, totals, prefix, count);
                } else {
                    p_cull_prefixlocal_kernel_smem<<<blocks, threads, threads * sizeof(unsigned int)>>>(remove, totals, prefix, count);
                }
                prefix_kernel<<<1, prefix_threads>>>(totals, blocks, remainder);
                p_cull_sum_scatter<<<blocks, threads>>>(particles, buffer, remove, prefix, totals, count);
            }
            check(cudaGetLastError());
            check(cudaEventRecord(stop));
            check(cudaEventSynchronize(stop));
            float gpu_ms = 0;
            check(cudaEventElapsedTime(&gpu_ms, start, stop));
            unsigned int tally = 0;
            check(cudaMemcpy(&tally, remainder, sizeof(tally), cudaMemcpyDeviceToHost));
            const std::size_t survivors = mode == 0 ? tally : count - tally;
            if(survivors != expected.size()) throw std::runtime_error("survivor count mismatch");
            std::vector<PredictionParticle> output(survivors);
            check(cudaMemcpy(output.data(), buffer, survivors * sizeof(PredictionParticle), cudaMemcpyDeviceToHost));
            const double wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wall_start).count();
            std::vector<std::size_t> ids;
            ids.reserve(survivors);
            for(const auto& p : output) ids.push_back(p.ID);
            if(mode == 0) std::sort(ids.begin(), ids.end());
            if(ids != expected) throw std::runtime_error("survivor IDs/order mismatch");
            if(trial >= 0) std::printf("%zu,%u,%d,%d,%zu,%.6f,%.6f\n", count, dead_percent, mode, trial, survivors, gpu_ms, wall_ms);
        }
    }
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(particles));
    check(cudaFree(buffer));
    check(cudaFree(remove));
    check(cudaFree(totals));
    check(cudaFree(prefix));
    check(cudaFree(remainder));
}
