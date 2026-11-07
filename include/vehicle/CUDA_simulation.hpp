#pragma once
#include <cstddef>
#include "vehicleBatch.hpp"
#include "environment.hpp"
#include "ProbabilityMap.hpp"
#include "vehicle/controller.hpp"
#include "vehicle/dynamics.hpp"
#include "vehicle/vehicleTypes.hpp"
#include "SearchPlanner.hpp"


class CUDA_simulation
{
public:
    CudaSimulation(SearchPlanner& planner, 
        DriftItem& target, 
        ProbabilityMap& p_map,
        environment& env,
        std::size_t N,
        float dt):
    planner_(planner),
    target_(target),
    p_map_(p_map),
    env_(env),
    N_(N),
    dt_(dt)
    {};
private:

    //init objects
    Controller controller;
    Dynamics dynamics;
    VehicleState vs{};
    Action action{};
    VehicleState newState{};
    SearchPlanner& planner_;
    DriftItem& target_;
    ProbabilityMap& p_map_;
    environment& env_;


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
    std::size_t N_;
    float dt_;

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    ~CUDA_simulation();

    void runCudaSimuation(
        std::size_t steps
    );
    
    void init_simulation(
        vehicleBatch& vehicles
    );

    void allocate_vehicles();
    void step_vehicles();

    __global__ void updateVehicles();

};


