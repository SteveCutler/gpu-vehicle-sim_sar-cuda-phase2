#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include "vehicle/vehicleBatch_CUDA.hpp"
#include "vehicle/environment_CUDA.hpp"
#include "vehicle/ProbabilityMap_CUDA.hpp"
#include "vehicle/controller_CUDA.hpp"
#include "vehicle/dynamics_CUDA.hpp"
#include "vehicle/vehicleTypes.hpp"
#include "vehicle/SearchPlanner_CUDA.hpp"


class CUDA_simulation
{
public:
    CUDA_simulation(SearchPlanner_CUDA& planner, 
        DriftItem& target, 
        ProbabilityMap_CUDA& p_map,
        ProbabilityMap_CUDA& temp_map,
        environment& env,
        std::size_t N,
        float dt):
        planner_(planner),
        target_(target),
        p_map_(p_map),
        temp_map_(temp_map),
        env_(env),
        v_count_(N),
        dt_(dt)
        {};
    
    CUDA_simulation(const CUDA_simulation&) = delete;
    CUDA_simulation& operator=(const CUDA_simulation&) = delete;

    ~CUDA_simulation();

    void runCudaSimulation(vehicleBatch_CUDA& vehicles, std::size_t steps);
    void init_simulation(vehicleBatch_CUDA& vehicles);
    void HtD_vehicles(const vehicleBatch_CUDA& vehicles);
    void DtH_vehicles(vehicleBatch_CUDA& vehicles);
    void step_vehicles();
    
    void HtD_fluid();
    void DtH_fluid();

    //init objects
    Controller controller;
    Dynamics dynamics;
    VehicleState vs{};
    Action action{};
    VehicleState newState{};
    SearchPlanner_CUDA& planner_;
    DriftItem& target_;
    ProbabilityMap_CUDA& p_map_;
    ProbabilityMap_CUDA& temp_map_;
    environment& env_;


private:



    //vehicle pointer vars
    float* vs_x = nullptr;
    float* vs_y = nullptr;
    float* vs_vx = nullptr;
    float* vs_vy = nullptr;
    float* vs_heading = nullptr;
    float* vs_turnRate = nullptr;
    float* vs_goalx = nullptr;
    float* vs_goaly = nullptr;

    //fluid pointer vars
    float* d_u_ = nullptr;
    float* d_v_ = nullptr;
    float* d_u_prev_ = nullptr;
    float* d_v_prev_ = nullptr;

    std::size_t cell_count_ = 0;
    std::size_t v_count_;
    float dt_;

};


