#pragma once
#include "vehicle/vehicleTypes.hpp"
#include "vehicle/ProbabilityMap.hpp"
#include "vehicle/environment.hpp"
#include "vehicle/dynamics.hpp"
#include "vehicle/controller.hpp"
#include <cuda_runtime.h>
#include <vector>

class SearchPlanner_CUDA
{
private:

    std::size_t horizon_;
    std::vector<std::vector<PredictionParticle>> forecast_;
    std::vector<std::pair<float,float>> wayPoints;
    std::vector<std::size_t> wpCount_;
    std::size_t w_index_;

public:
    SearchPlanner_CUDA(std::size_t horizon);
    
    std::pair<float,float> chooseWaypoint(VehicleState vs, 
        const std::vector<PredictionParticle>& particles, 
        const environment& env, 
        const float dt,
        ProbabilityMap& tempMap,
        Dynamics& dynamics,
        Controller& controller,
        std::size_t height_);
    ~SearchPlanner_CUDA() = default;
};
