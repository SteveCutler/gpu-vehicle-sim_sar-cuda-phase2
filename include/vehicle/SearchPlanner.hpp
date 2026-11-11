#pragma once
#include "vehicle/vehicleTypes.hpp"
#include "vehicle/ProbabilityMap.hpp"
#include "vehicle/environment.hpp"
#include "vehicle/dynamics.hpp"
#include "vehicle/controller.hpp"
#include <vector>
#include "vehicle/routeData.hpp"

class SearchPlanner
{
    friend class CUDA_simulation;
private:

    std::size_t horizon_;
    std::vector<std::vector<PredictionParticle>> forecast_;
    std::vector<std::size_t> wpCount_;
    std::size_t w_index_;
    
    public:
    SearchPlanner(std::size_t horizon, std::size_t v_num);
    
    std::pair<float,float> chooseWaypoint(VehicleState vs, 
    const std::vector<PredictionParticle>& particles, 
    const environment& env, 
    const float dt,
    ProbabilityMap& tempMap,
    Dynamics& dynamics,
    Controller& controller,
    std::size_t height_);
    ~SearchPlanner() = default;

    std::vector<std::pair<float,float>> wayPoints;
    PlannerPreview preview;
};
