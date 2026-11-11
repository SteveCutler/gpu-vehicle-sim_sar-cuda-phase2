#pragma once
#include <vector>
#include <random>
#include "vehicle/DriftItem.hpp"
#include "vehicle/environment.hpp"
#include "vehicle/DriftDynamics.hpp"
#include "vehicle/vehicleTypes.hpp"

class ProbabilityMap
{
    // The CUDA backend uploads/downloads the existing CPU state.
    friend class CUDA_simulation;

private:
DriftDynamics DD_;
std::mt19937 rng_;
float p_rad_;
float origin_x_;
float origin_y_;

public:
    std::size_t width_;
    std::size_t height_;
    std::size_t target_cell;
    std::vector<PredictionParticle> particles_;
    std::vector<PredictionParticle> particle_buffer;
    std::size_t cell_size_;
    std::vector<unsigned int> map_;
    std::size_t num_p_;
    std::vector<std::size_t>occupied_cells;
    
public:
    ProbabilityMap(DriftDynamics& DD, ::size_t width, std::size_t height, std::size_t num_p, float est_x, float est_y, float p_rad, std::size_t cell_size);

    void update(const environment& env, float dt);
    
    void add_Particles();

    void computeProb();

    std::pair<float,float> convertCell(std::size_t cell);

    void updateGoal(VehicleState& vs);

    void particleScan(VehicleState& vs);

    ~ProbabilityMap() = default;
};
