#include "vehicle/CUDA_simulation.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

void fluidCheck(cudaError_t error) {
    if(error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}
#define FLUID_CHECK(call) fluidCheck(call)

__device__ float clamp_value(float value, float low, float high) {
    return fminf(high, fmaxf(low, value));
}

__device__ float density_sample(std::size_t i, const float *m_density_r, const float *m_density_g,
                                const float *m_density_b) {
    const float density = fmaxf(m_density_r[i], fmaxf(m_density_g[i], m_density_b[i]));
    return clamp_value(density, 0.f, 1.f);
}

__global__ void calc_vel_kernel(const float *m_noise, float *m_u_velocity, float *m_v_velocity,
                                float m_curl_mult, float dt, std::size_t m_width,
                                std::size_t m_height) {
    // One thread per grid cell.
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= m_width * m_height) return;
    std::size_t x = index % m_width;
    std::size_t y = index / m_width;
    if(y == 0 || y >= m_height - 1 || x == 0 || x >= m_width - 1) return;

    float xl = m_noise[index - 1];
    float xr = m_noise[index + 1];
    float yt = m_noise[index - m_width];
    float yb = m_noise[index + m_width];

    float dx = (xr - xl) / 2;
    float dy = (yb - yt) / 2;

    m_u_velocity[index] += (-dy * m_curl_mult * dt);
    m_v_velocity[index] += (dx * m_curl_mult * dt);
}

__global__ void advect_vel_kernel(const float *m_u_velocity_prev, const float *m_v_velocity_prev,
                                  float *m_u_velocity, float *m_v_velocity,
                                  const float *m_density_r, const float *m_density_g,
                                  const float *m_density_b, float m_vel_decay, float m_drift,
                                  float m_buoyancy, float dt, std::size_t m_width,
                                  std::size_t m_height) {
    // One thread per grid cell.
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= m_width * m_height) return;
    std::size_t x = index % m_width;
    std::size_t y = index / m_width;
    if(y == 0 || y >= m_height - 1) return;
    std::size_t row = y * m_width;

    // addition calculations are cheap for CPU
    std::size_t i = row + x;

    // index velocity
    float old_u_vel = m_u_velocity_prev[i];
    float old_v_vel = m_v_velocity_prev[i];

    // removed modulo and multiplicatin operations happening for every cell

    // backwards location lookup
    float new_u_pos = x - old_u_vel * dt;
    float new_v_pos = y - old_v_vel * dt;

    // finding 4 corners for bilinear interpolation, clamping in boundaries
    float x_sample = clamp_value(new_u_pos, 1.0f, m_width - 2.0f);
    float y_sample = clamp_value(new_v_pos, 1.0f, m_height - 2.0f);

    std::size_t x_floor = floorf(x_sample);
    std::size_t x_high = x_floor + 1;
    std::size_t y_floor = floorf(y_sample);
    std::size_t y_high = y_floor + 1;

    // finding values at corners
    std::size_t tl_index = x_floor + y_floor * m_width;
    std::size_t tr_index = x_high + y_floor * m_width;
    std::size_t bl_index = x_floor + y_high * m_width;
    std::size_t br_index = x_high + y_high * m_width;

    float tl_u = m_u_velocity_prev[tl_index];
    float tr_u = m_u_velocity_prev[tr_index];
    float bl_u = m_u_velocity_prev[bl_index];
    float br_u = m_u_velocity_prev[br_index];

    float tl_v = m_v_velocity_prev[tl_index];
    float tr_v = m_v_velocity_prev[tr_index];
    float bl_v = m_v_velocity_prev[bl_index];
    float br_v = m_v_velocity_prev[br_index];

    // distance from edges
    float dx = x_sample - x_floor;
    float dy = y_sample - y_floor;

    // weighting calculations
    float tl_weight = (1 - dx) * (1 - dy);
    float tr_weight = dx * (1 - dy);
    float bl_weight = (1 - dx) * dy;
    float br_weight = dx * dy;

    float new_u_vel =
        (tl_u * tl_weight + tr_u * tr_weight + bl_u * bl_weight + br_u * br_weight) * m_vel_decay;
    float new_v_vel =
        (tl_v * tl_weight + tr_v * tr_weight + bl_v * bl_weight + br_v * br_weight) * m_vel_decay;
    new_u_vel -= m_drift * density_sample(i, m_density_r, m_density_g, m_density_b) * dt;
    new_v_vel -= m_buoyancy * density_sample(i, m_density_r, m_density_g, m_density_b) * dt;

    m_u_velocity[i] = new_u_vel;
    m_v_velocity[i] = new_v_vel;
}

__global__ void diffuse_vel_kernel(const float *m_u_velocity_prev, const float *m_v_velocity_prev,
                                   float *m_u_velocity, float *m_v_velocity, float m_viscosity,
                                   float dt, std::size_t m_width, std::size_t m_height) {
    // One thread per grid cell.
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= m_width * m_height) return;
    std::size_t x = index % m_width;
    std::size_t y = index / m_width;
    if(y == 0 || y >= m_height - 1 || x == 0 || x >= m_width - 1) return;
    std::size_t row = y * m_width;

    // calculate indexx
    std::size_t i = row + x;

    // calc corner indices

    std::size_t l_index = i - 1;
    std::size_t r_index = i + 1;
    std::size_t u_index = i - m_width;
    std::size_t d_index = i + m_width;

    // getting data for laplacian
    float left_u = m_u_velocity_prev[l_index];
    float right_u = m_u_velocity_prev[r_index];
    float up_u = m_u_velocity_prev[u_index];
    float down_u = m_u_velocity_prev[d_index];
    float center_u = m_u_velocity_prev[i];

    float left_v = m_v_velocity_prev[l_index];
    float right_v = m_v_velocity_prev[r_index];
    float up_v = m_v_velocity_prev[u_index];
    float down_v = m_v_velocity_prev[d_index];
    float center_v = m_v_velocity_prev[i];

    // calculate laplacian
    float lap_u = left_u + right_u + up_u + down_u - (4 * center_u);
    float lap_v = left_v + right_v + up_v + down_v - (4 * center_v);

    // calculate and write new density
    float new_u_vel = m_u_velocity_prev[i] + m_viscosity * lap_u * dt;
    float new_v_vel = m_v_velocity_prev[i] + m_viscosity * lap_v * dt;

    m_u_velocity[i] = new_u_vel;
    m_v_velocity[i] = new_v_vel;
}

__global__ void divergence_kernel(const float *m_u_velocity, const float *m_v_velocity,
                                  float *m_divergence, std::size_t m_width, std::size_t m_height) {
    // One thread per grid cell.
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= m_width * m_height) return;
    std::size_t x = index % m_width;
    std::size_t y = index / m_width;
    if(y == 0 || y >= m_height - 1 || x == 0 || x >= m_width - 1) return;

    float xl = m_u_velocity[index - 1];
    float xr = m_u_velocity[index + 1];
    float yt = m_v_velocity[index - m_width];
    float yb = m_v_velocity[index + m_width];

    float div = ((xr - xl) * .5) + ((yb - yt) * .5);

    m_divergence[index] = div;
}

__global__ void pressure_kernel(const float *m_pressure_prev, const float *m_divergence,
                                float *m_pressure, std::size_t m_width, std::size_t m_height) {
    // One thread per grid cell.
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= m_width * m_height) return;
    std::size_t x = index % m_width;
    std::size_t y = index / m_width;
    if(y == 0 || y >= m_height - 1 || x == 0 || x >= m_width - 1) return;

    float xl = m_pressure_prev[index - 1];
    float xr = m_pressure_prev[index + 1];
    float yt = m_pressure_prev[index - m_width];
    float yb = m_pressure_prev[index + m_width];

    float pressure = (xl + xr + yt + yb - m_divergence[index]) * 0.25f;

    m_pressure[index] = pressure;
}

__global__ void project_kernel(const float *m_pressure, float *m_u_velocity, float *m_v_velocity,
                               std::size_t m_width, std::size_t m_height) {
    // One thread per grid cell.
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= m_width * m_height) return;
    std::size_t x = index % m_width;
    std::size_t y = index / m_width;
    if(y == 0 || y >= m_height - 1 || x == 0 || x >= m_width - 1) return;

    float xl = m_pressure[index - 1];
    float xr = m_pressure[index + 1];
    float yt = m_pressure[index - m_width];
    float yb = m_pressure[index + m_width];

    float pres_x = (xr - xl) / 2.f;
    float pres_y = (yb - yt) / 2.f;

    m_u_velocity[index] -= pres_x;
    m_v_velocity[index] -= pres_y;
}

__global__ void diffuse_kernel(float a, const float *source, const float *current, float *next,
                               std::size_t m_width, std::size_t m_height) {
    // One thread per grid cell.
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= m_width * m_height) return;
    std::size_t x = index % m_width;
    std::size_t y = index / m_width;
    if(y == 0 || y >= m_height - 1 || x == 0 || x >= m_width - 1) return;
    std::size_t row = y * m_width;

    const float denominator = 1.f + 4.f * a;

    const std::size_t i = row + x;

    const float neighbourSum =
        current[i - 1] + current[i + 1] + current[i - m_width] + current[i + m_width];

    next[i] = (source[i] + a * neighbourSum) / denominator;
}

__global__ void advect_decay_kernel(const float *m_u_velocity, const float *m_v_velocity,
                                    const float *source, float *dest, float m_decay, float dt,
                                    std::size_t m_width, std::size_t m_height) {
    // One thread per grid cell.
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= m_width * m_height) return;
    std::size_t x = index % m_width;
    std::size_t y = index / m_width;
    if(y == 0 || y >= m_height - 1) return;
    std::size_t row = y * m_width;

    std::size_t i = row + x;
    // index velocity
    float v_vel = m_v_velocity[i];
    float u_vel = m_u_velocity[i];

    // backwards location lookup
    float new_x = clamp_value((x - u_vel * dt), 1.f, m_width - 2.f);
    float new_y = clamp_value((y - v_vel * dt), 1.f, m_height - 2.f);

    // finding 4 corners for bilinear interpolation, clamping in boundaries
    std::size_t x_low = floorf(new_x);
    std::size_t x_high = x_low + 1;
    std::size_t y_low = floorf(new_y);
    std::size_t y_high = y_low + 1;

    // finding values at corners
    float tl = source[x_low + y_low * m_width];
    float tr = source[x_high + y_low * m_width];
    float bl = source[x_low + y_high * m_width];
    float br = source[x_high + y_high * m_width];

    // distance from edges
    float dx = new_x - x_low;
    float dy = new_y - y_low;

    // weighting calculations
    float tl_weight = (1 - dx) * (1 - dy);
    float tr_weight = dx * (1 - dy);
    float bl_weight = (1 - dx) * dy;
    float br_weight = dx * dy;

    float new_dens = tl * tl_weight + tr * tr_weight + bl * bl_weight + br * br_weight;

    dest[i] = new_dens * m_decay;
}

__global__ void vel_boundaries_y_kernel(float *m_u_velocity, float *m_v_velocity,
                                        std::size_t m_width, std::size_t m_height) {
    std::size_t y = blockIdx.x * blockDim.x + threadIdx.x;
    if(y >= m_height) return;
    std::size_t b_l = y * m_width;
    std::size_t b_r = (m_width - 1) + y * m_width;

    // horizontal vel 0
    m_u_velocity[b_l] = 0.f;
    m_u_velocity[b_r] = 0.f;
    // vertical vel refers to neighbour
    m_v_velocity[b_l] = m_v_velocity[b_l + 1];
    m_v_velocity[b_r] = m_v_velocity[b_r - 1];
}

__global__ void vel_boundaries_x_kernel(float *m_u_velocity, float *m_v_velocity,
                                        std::size_t m_width, std::size_t m_height) {
    std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= m_width) return;
    std::size_t b_t = x;
    std::size_t b_b = x + (m_height - 1) * m_width;

    // vertical vel refers to neighbour
    m_u_velocity[b_t] = m_u_velocity[b_t + m_width];
    m_u_velocity[b_b] = m_u_velocity[b_b - m_width];
    // vertical vel 0
    m_v_velocity[b_t] = 0.f;
    m_v_velocity[b_b] = 0.f;
}

__global__ void pressure_boundaries_y_kernel(float *m_pressure, std::size_t m_width,
                                             std::size_t m_height) {
    std::size_t y = blockIdx.x * blockDim.x + threadIdx.x;
    if(y >= m_height) return;
    std::size_t b_l = y * m_width;
    std::size_t b_r = (m_width - 1) + y * m_width;

    // pressure refers to x
    m_pressure[b_l] = m_pressure[b_l + 1];
    m_pressure[b_r] = m_pressure[b_r - 1];
}

__global__ void pressure_boundaries_x_kernel(float *m_pressure, std::size_t m_width,
                                             std::size_t m_height) {
    std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= m_width) return;
    std::size_t b_t = x;
    std::size_t b_b = x + (m_height - 1) * m_width;

    // refers to vertical neighbour
    m_pressure[b_t] = m_pressure[b_t + m_width];
    m_pressure[b_b] = m_pressure[b_b - m_width];
}

__global__ void scalar_boundaries_y_kernel(float *field, std::size_t m_width,
                                           std::size_t m_height) {
    std::size_t y = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if(y >= m_height - 1) return;
    const std::size_t row = y * m_width;
    field[row] = field[row + 1];
    field[row + m_width - 1] = field[row + m_width - 2];
}

__global__ void scalar_boundaries_x_kernel(float *field, std::size_t m_width,
                                           std::size_t m_height) {
    std::size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if(x >= m_width - 1) return;
    field[x] = field[x + m_width];
    field[(m_height - 1) * m_width + x] = field[(m_height - 2) * m_width + x];
}

__global__ void scalar_corners_kernel(float *field, std::size_t m_width, std::size_t m_height) {
    // A separate launch makes all edge writes visible before reading corners.
    // top left corner
    field[0] = (field[1] + field[m_width]) * .5f;

    // top right corner
    field[m_width - 1] = (field[(m_width * 2 - 1)] + field[m_width - 2]) * .5f;

    // bottom left corner
    field[m_width * (m_height - 1)] =
        (field[m_width * (m_height - 1) + 1] + field[m_width * (m_height - 2)]) * .5f;

    // bottom right corner
    field[m_width * m_height - 1] =
        (field[m_width * m_height - 2] + field[m_width * m_height - m_width - 1]) * .5f;
}

__global__ void add_source_kernel(float *m_density_r, float *m_density_g, float *m_density_b,
                                  std::size_t x, std::size_t y, float size, Vec3 clr,
                                  std::size_t m_width, std::size_t m_height) {
    // Same integer offsets as the CPU loops, including fractional radii.
    int first = static_cast<int>(-size);
    int last = static_cast<int>(size);
    std::size_t side = last - first + 1;
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= side * side) return;
    int dx = first + static_cast<int>(index % side);
    int dy = first + static_cast<int>(index / side);
    std::size_t new_y = y + dy;
    std::size_t row = new_y * m_width;
    // new_x
    std::size_t new_x = x + dx;

    if(new_x > 1 && new_x < m_width - 1 && new_y > 1 && new_y < m_height - 1) {

        float dist = (dx * dx) + (dy * dy);
        float rad_sqr = size * size;

        if(dist < rad_sqr) {
            float density_r = clr.x * (rad_sqr - dist) / rad_sqr;
            float density_g = clr.y * (rad_sqr - dist) / rad_sqr;
            float density_b = clr.z * (rad_sqr - dist) / rad_sqr;

            size_t pos = new_x + row;
            m_density_r[pos] = clamp_value((m_density_r[pos] + density_r), 0.f, 1.f);
            m_density_g[pos] = clamp_value((m_density_g[pos] + density_g), 0.f, 1.f);
            m_density_b[pos] = clamp_value((m_density_b[pos] + density_b), 0.f, 1.f);
        }
    }
}

} // namespace

void CUDA_simulation::calc_noise(float dt) {
    FluidGrid &fluid = env_.fluid_grid;
    // Keep the existing FastNoiseLite implementation for this baseline.
    fluid.calcNoise(dt);
    FLUID_CHECK(cudaMemcpy(d_noise_, fluid.m_noise.data(),
                           env_.m_width * env_.m_height * sizeof(float), cudaMemcpyHostToDevice));
}

void CUDA_simulation::calc_vel(float dt) {
    FluidGrid &fluid = env_.fluid_grid;
    std::size_t width = env_.m_width;
    std::size_t height = env_.m_height;
    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;
    calc_vel_kernel<<<blocks, threads>>>(d_noise_, d_u_, d_v_, fluid.m_curl_mult, dt, width,
                                         height);
    FLUID_CHECK(cudaGetLastError());
}

void CUDA_simulation::advect_vel(float dt) {
    FluidGrid &fluid = env_.fluid_grid;
    std::size_t width = env_.m_width;
    std::size_t height = env_.m_height;
    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;
    advect_vel_kernel<<<blocks, threads>>>(d_u_prev_, d_v_prev_, d_u_, d_v_, d_density_r_,
                                           d_density_g_, d_density_b_, fluid.m_vel_decay,
                                           fluid.m_drift, fluid.m_buoyancy, dt, width, height);
    FLUID_CHECK(cudaGetLastError());
}

void CUDA_simulation::diffuse_vel(float dt) {
    FluidGrid &fluid = env_.fluid_grid;
    std::size_t width = env_.m_width;
    std::size_t height = env_.m_height;
    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;
    diffuse_vel_kernel<<<blocks, threads>>>(d_u_prev_, d_v_prev_, d_u_, d_v_, fluid.m_viscosity, dt,
                                            width, height);
    FLUID_CHECK(cudaGetLastError());
}

void CUDA_simulation::swap_vel() {
    std::swap(d_u_, d_u_prev_);
    std::swap(d_v_, d_v_prev_);
}

void CUDA_simulation::swap_density() {
    std::swap(d_density_r_, d_density_r_prev_);
    std::swap(d_density_g_, d_density_g_prev_);
    std::swap(d_density_b_, d_density_b_prev_);
}

void CUDA_simulation::vel_boundaries() {
    std::size_t width = env_.m_width;
    std::size_t height = env_.m_height;
    int threads = 256;
    // Keep left/right before top/bottom, as in the CPU implementation.
    vel_boundaries_y_kernel<<<(height + threads - 1) / threads, threads>>>(d_u_, d_v_, width,
                                                                           height);
    FLUID_CHECK(cudaGetLastError());
    vel_boundaries_x_kernel<<<(width + threads - 1) / threads, threads>>>(d_u_, d_v_, width,
                                                                          height);
    FLUID_CHECK(cudaGetLastError());
}

void CUDA_simulation::pressure_boundaries() {
    std::size_t width = env_.m_width;
    std::size_t height = env_.m_height;
    int threads = 256;
    // Keep left/right before top/bottom, as in the CPU implementation.
    pressure_boundaries_y_kernel<<<(height + threads - 1) / threads, threads>>>(d_pressure_, width,
                                                                                height);
    FLUID_CHECK(cudaGetLastError());
    pressure_boundaries_x_kernel<<<(width + threads - 1) / threads, threads>>>(d_pressure_, width,
                                                                               height);
    FLUID_CHECK(cudaGetLastError());
}

void CUDA_simulation::scalar_boundaries(float *field) {
    std::size_t width = env_.m_width;
    std::size_t height = env_.m_height;
    int threads = 256;
    // Keep left/right before top/bottom, as in the CPU implementation.
    scalar_boundaries_y_kernel<<<(height + threads - 1) / threads, threads>>>(field, width, height);
    FLUID_CHECK(cudaGetLastError());
    scalar_boundaries_x_kernel<<<(width + threads - 1) / threads, threads>>>(field, width, height);
    FLUID_CHECK(cudaGetLastError());
    scalar_corners_kernel<<<1, 1>>>(field, width, height);
    FLUID_CHECK(cudaGetLastError());
}


void CUDA_simulation::project_step_tiled() {
    FluidGrid &fluid = env_.fluid_grid;
    std::size_t width = env_.m_width;
    std::size_t height = env_.m_height;
    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;
    divergence_kernel<<<blocks, threads>>>(d_u_, d_v_, d_divergence_, width, height);
    FLUID_CHECK(cudaGetLastError());

    // Preserve the CPU pressure warm start, iteration count and final swap.
    for(std::size_t k = 0; k < fluid.m_pressure_iter; k++) {
        pressure_kernel<<<blocks, threads>>>(d_pressure_prev_, 
                                            d_divergence_, 
                                            d_pressure_, 
                                            width,
                                            height);
        FLUID_CHECK(cudaGetLastError());
        pressure_boundaries();
        std::swap(d_pressure_, d_pressure_prev_);
    }
    std::swap(d_pressure_, d_pressure_prev_);
    pressure_boundaries();

    project_kernel<<<blocks, threads>>>(d_pressure_, d_u_, d_v_, width, height);
    FLUID_CHECK(cudaGetLastError());
    vel_boundaries();
}

void CUDA_simulation::project_step_global() {
    FluidGrid &fluid = env_.fluid_grid;
    std::size_t width = env_.m_width;
    std::size_t height = env_.m_height;
    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;
    divergence_kernel<<<blocks, threads>>>(d_u_, 
                                            d_v_, 
                                            d_divergence_, 
                                            width, 
                                            height);
    FLUID_CHECK(cudaGetLastError());

    // Preserve the CPU pressure warm start, iteration count and final swap.
    for(std::size_t k = 0; k < fluid.m_pressure_iter; k++) {
        pressure_kernel<<<blocks, threads>>>(d_pressure_prev_, d_divergence_, d_pressure_, width,
                                             height);
        FLUID_CHECK(cudaGetLastError());
        pressure_boundaries();
        std::swap(d_pressure_, d_pressure_prev_);
    }
    std::swap(d_pressure_, d_pressure_prev_);
    pressure_boundaries();

    project_kernel<<<blocks, threads>>>(d_pressure_, d_u_, d_v_, width, height);
    FLUID_CHECK(cudaGetLastError());
    vel_boundaries();
}

void CUDA_simulation::add_source(std::size_t x, std::size_t y, float size, Vec3 clr) {
    if(size < 0.f) return;
    int first = static_cast<int>(-size);
    int last = static_cast<int>(size);
    std::size_t side = last - first + 1;
    int threads = 256;
    int blocks = (side * side + threads - 1) / threads;
    add_source_kernel<<<blocks, threads>>>(d_density_r_, d_density_g_, d_density_b_, x, y, size,
                                           clr, env_.m_width, env_.m_height);
    FLUID_CHECK(cudaGetLastError());
}

void CUDA_simulation::diffuse(float dt, const float *source, float *&destination) {
    FluidGrid &fluid = env_.fluid_grid;
    std::size_t width = env_.m_width;
    std::size_t height = env_.m_height;
    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;
    const float a = fluid.m_diff_co * dt;
    // Take the source field as the initial guess, just like destination = source.
    FLUID_CHECK(
        cudaMemcpy(destination, source, width * height * sizeof(float), cudaMemcpyDeviceToDevice));
    float *current = destination;
    float *next = d_diffusion_scratch_;
    for(std::size_t iteration = 0; iteration < fluid.m_diffusion_iterations; iteration++) {
        diffuse_kernel<<<blocks, threads>>>(a, source, current, next, width, height);
        FLUID_CHECK(cudaGetLastError());
        scalar_boundaries(next);
        std::swap(current, next);
    }
    if(current != destination) {
        std::swap(destination, d_diffusion_scratch_);
    }
}

void CUDA_simulation::advect_decay(float dt, const float *source, float *dest) {
    FluidGrid &fluid = env_.fluid_grid;
    std::size_t width = env_.m_width;
    std::size_t height = env_.m_height;
    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;
    advect_decay_kernel<<<blocks, threads>>>(d_u_, d_v_, source, dest, fluid.m_decay, dt, width,
                                             height);
    FLUID_CHECK(cudaGetLastError());
}

void CUDA_simulation::step_fluid(float dt) {
    FluidGrid &fluid = env_.fluid_grid;
    fluid.m_elapsed += dt;

    // Same order as FluidGrid::update(). Kernels use the default stream.
    calc_noise(dt);
    calc_vel(dt);

    swap_vel();
    advect_vel(dt);
    vel_boundaries();

    swap_vel();
    diffuse_vel(dt);
    vel_boundaries();

    if (config_.pressure == PressureMode::Tiled) {
        project_step_tiled();
    } else {
        project_step_global();
    }

    for(auto &emitter : fluid.m_emitters) {
        add_source(emitter->m_pos.x, emitter->m_pos.y, emitter->m_rad, emitter->m_clr);
    }

    swap_density();
    diffuse(dt, d_density_r_prev_, d_density_r_);
    diffuse(dt, d_density_g_prev_, d_density_g_);
    diffuse(dt, d_density_b_prev_, d_density_b_);

    swap_density();
    advect_decay(dt, d_density_r_prev_, d_density_r_);
    advect_decay(dt, d_density_g_prev_, d_density_g_);
    advect_decay(dt, d_density_b_prev_, d_density_b_);
    scalar_boundaries(d_density_r_);
    scalar_boundaries(d_density_g_);
    scalar_boundaries(d_density_b_);

    // Target/particle drift still samples the CPU velocity arrays.
    // The blocking download also reports asynchronous kernel failures.
    DtH_fluid();
#ifdef ENABLE_RENDERER
    fluid.gen_pixels_Rows(0, fluid.m_height);
#endif
}
