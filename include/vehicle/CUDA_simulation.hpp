#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include "vehicle/vehicleBatch.hpp"
#include "vehicle/environment.hpp"
#include "vehicle/ProbabilityMap.hpp"
#include "vehicle/controller.hpp"
#include "vehicle/dynamics.hpp"
#include "vehicle/vehicleTypes.hpp"
#include "vehicle/SearchPlanner.hpp"
#include "vehicle/DriftItem.hpp"


class CUDA_simulation
{
public:
    CUDA_simulation(SearchPlanner& planner,
        DriftItem& target, 
        ProbabilityMap& p_map,
        ProbabilityMap& temp_map,
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

    std::size_t runCudaSimulation(vehicleBatch& vehicles, std::size_t steps);
    void init_simulation(vehicleBatch& vehicles);
    void HtD_vehicles(const vehicleBatch& vehicles);
    void DtH_vehicles(vehicleBatch& vehicles);
    void step_vehicles();
    void calc_p_horizon();
    
    void HtD_fluid();
    void DtH_fluid();

    void particle_scan(std::size_t steps);

    

    //init objects
    Controller controller;
    Dynamics dynamics;
    VehicleState vs{};
    Action action{};
    VehicleState newState{};
    SearchPlanner& planner_;
    DriftItem& target_;
    ProbabilityMap& p_map_;
    ProbabilityMap& temp_map_;
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
    float sensor_radius_ = 0;

    //fluid pointer vars
    float* d_u_ = nullptr;
    float* d_v_ = nullptr;
    float* d_u_prev_ = nullptr;
    float* d_v_prev_ = nullptr;

    //particle device pointer
    PredictionParticle* d_particles_ = nullptr;
    unsigned char* d_remove_particles_ = nullptr;
    PredictionParticle* d_p_horizons = nullptr;

    std::size_t cell_count_ = 0;
    std::size_t v_count_;
    float dt_;

};
