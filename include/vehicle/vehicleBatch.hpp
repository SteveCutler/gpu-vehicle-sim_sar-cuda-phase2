#pragma once
#include <vector>
#include <cstddef>
#include "vehicle/environment.hpp"
#include "vehicle/vehicleTypes.hpp"
#include "vehicle/ProbabilityMap.hpp"


class vehicleBatch
{
    // The CUDA backend uploads/downloads the existing CPU state.
    friend class CUDA_simulation;


private:
    std::vector<float> m_x, m_y;
    std::vector<float> m_vx, m_vy;
    std::vector<float> m_heading, m_turnRate;
    std::size_t m_size;
    std::vector<float> m_sensor_rad;
    float init_goalx;
    float init_goaly;
    
    
    public:
    std::vector<float> m_goalx, m_goaly;
    
    vehicleBatch(std::size_t N, environment env, float goal_x, float goal_y);

    VehicleState load(std::size_t i) const;

    void set(std::size_t i, const VehicleState& state);
    
    std::size_t getSize() const;

    bool scan(VehicleState& vs, ProbabilityMap& p_map, DriftItem& target);

};


// TO DO write update function
