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
    float init_goalx;
    float init_goaly;
    
    
    public:
    std::vector<float> m_sensor_rad;
    std::vector<float> m_goalx, m_goaly;
    
    vehicleBatch(std::size_t N, environment env, float goal_x, float goal_y);

    VehicleState load(std::size_t i) const;

    bool targetCheck(const VehicleState& vs, const DriftItem& target, std::size_t height) const;
    void goalCheck(VehicleState& vs, ProbabilityMap& p_map);

    void set(std::size_t i, const VehicleState& state);
    
    std::size_t getSize() const;

    bool scan(VehicleState& vs, ProbabilityMap& p_map, DriftItem& target);

};


// TO DO write update function
