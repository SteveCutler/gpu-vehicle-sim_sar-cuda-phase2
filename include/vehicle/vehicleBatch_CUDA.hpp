#pragma once
#include <vector>
#include <cstddef>
#include "vehicle/environment_CUDA.hpp"
#include "vehicle/vehicleTypes.hpp"
#include "vehicle/ProbabilityMap_CUDA.hpp"


class vehicleBatch_CUDA
{

private:
std::size_t m_size;
std::vector<float> m_sensor_rad;
float init_goalx;
float init_goaly;


public:
    std::vector<float> m_x, m_y;
    std::vector<float> m_vx, m_vy;
    std::vector<float> m_heading, m_turnRate;
    std::vector<float> m_goalx, m_goaly;
    
    vehicleBatch_CUDA(std::size_t N, environment_CUDA env, float goal_x, float goal_y);

    VehicleState load(std::size_t i) const;

    void set(std::size_t i, const VehicleState& state);
    
    std::size_t getSize() const;

    bool scan(VehicleState& vs, ProbabilityMap_CUDA& p_map, DriftItem& target);

};


// TO DO write update function

