#include <vector>
#include "vehicle/CUDA_simulation.hpp"



//cuda error check function, wrapping all mallocs and transfers with it
inline void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}
#define CUDA_CHECK(call) cudaCheck(call)


///


__global__ void diffuse_velocity_kernel(
    const float* u_prev, const float* v_prev,
    float* u, float* v,
    std::size_t width, std::size_t height, float dt)
{
    // One thread calculates one output cell.
}


void CUDA_simulation::calc_noise_vel(float dt){
    FluidGrid& fluid = env_.fluid_grid;
    std::size_t width = env_.m_width;
    std::size_t height = env_.m_height;
    std::size_t bytes = width * height * sizeof(float);

    //cpu function updates noise field, 
    //simpler than porting the noise generation algorithm to CUDA
    fluid.calcNoise(dt_);

    cudaCheck(cudaMemcpy(
        d_noise_, fluid.m_noise.data(), bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;

    calc_vel_kernel<<<blocks, threads>>>(
        d_noise_, d_u_, d_v_, width, height, fluid.m_curl_mult, dt_);

    cudaCheck(cudaGetLastError());
};


//PORTING FUNCTIONS FROM CPU FLUID SOLVER OVER
void CUDA_simulation::step_fluid(float dt)
{
    fluid.m_elapsed += dt;

    calc_noise_vel(dt);

    //to port:
    //calc noise
    //calc vel
    //advect vel
    //vel boundaries
    //swap vel
    //diffuse vel
    //project step
    //add source
    //swap density
    //diffuse r/g/b
    //advect decay
    //scalar boundaries density r g b
    //


    
    diffuse_velocity_kernel<<<blocks, threads>>>(
        d_u_prev_, d_v_prev_, d_u_, d_v_,
        env_.m_width, env_.m_height, dt);
}