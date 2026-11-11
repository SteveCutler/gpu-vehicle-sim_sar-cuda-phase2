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
        float dt,
        std::size_t cell_size):
        planner_(planner),
        target_(target),
        p_map_(p_map),
        temp_map_(temp_map),
        env_(env),
        v_count_(N),
        dt_(dt),
        cell_size_(cell_size)
        {};
    
    CUDA_simulation(const CUDA_simulation&) = delete;
    CUDA_simulation& operator=(const CUDA_simulation&) = delete;

    ~CUDA_simulation();

    std::size_t runCudaSimulation(vehicleBatch& vehicles, std::size_t steps);
    void init_simulation(vehicleBatch& vehicles);
    void HtD_vehicles(const vehicleBatch& vehicles);
    void DtH_vehicles(vehicleBatch& vehicles);

    void step_vehicles();

    void calc_p_horizon(std::size_t steps);
    void particle_scan();
    void find_waypoints();
    void calc_occ_maps(std::size_t steps, std::size_t p_count);
    void calc_paths(vehicleBatch& vehicles, std::size_t steps, std::size_t p_count);
    void score_paths(vehicleBatch& vehicles);
    std::pair<float,float> convert_cell(std::size_t cell);

    void step_fluid(float dt);
    void HtD_fluid();
    void DtH_fluid();

    void calc_noise(float dt);
    void calc_vel(float dt);
    void advect_vel(float dt);
    void diffuse_vel(float dt);
    void swap_vel();
    void swap_density();
    void vel_boundaries();
    void pressure_boundaries();
    void scalar_boundaries(float* field);
    void project_step();
    void add_source(std::size_t x, std::size_t y, float size, Vec3 clr);
    void diffuse(float dt, const float* source, float*& destination);
    void advect_decay(float dt, const float* source, float* dest);


    

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



    //VEHICLE POINTER VARS
    float* vs_x = nullptr;
    float* vs_y = nullptr;
    float* vs_vx = nullptr;
    float* vs_vy = nullptr;
    float* vs_heading = nullptr;
    float* vs_turnRate = nullptr;
    float* vs_goalx = nullptr;
    float* vs_goaly = nullptr;
    float sensor_radius_ = 0;

    //FLUID POINTER VARS
    // Noise used to generate the velocity disturbance
    float* d_noise_ = nullptr;
    float elapsed_dt = 0.f;

    // density
    float* d_density_r_ = nullptr;
    float* d_density_g_ = nullptr;
    float* d_density_b_ = nullptr;
    float* d_density_r_prev_ = nullptr;
    float* d_density_g_prev_ = nullptr;
    float* d_density_b_prev_ = nullptr;

    // Projection
    float* d_divergence_ = nullptr;
    float* d_pressure_ = nullptr;
    float* d_pressure_prev_ = nullptr;
    float* d_diffusion_scratch_ = nullptr;

    float* d_u_ = nullptr;
    float* d_v_ = nullptr;
    float* d_u_prev_ = nullptr;
    float* d_v_prev_ = nullptr;

    //WAYPOINT DEVICES
    float* d_waypoints_x = nullptr;
    float* d_waypoints_y = nullptr;
    unsigned int* d_occupancy_maps_ = nullptr;
    std::vector<unsigned int> occupancy_maps_;
    std::size_t candidate_waypoints_ = 3;
    unsigned int* d_best_wp = nullptr;

    //PARTICLE POINTER VARS
    PredictionParticle* d_particles_ = nullptr;
    unsigned char* d_remove_particles_ = nullptr;
    PredictionParticle* d_p_horizons = nullptr;
    std::vector<PredictionParticle> horizons_;
    unsigned char* d_hit_map = nullptr;

    std::size_t cell_count_;
    std::size_t v_count_;
    float dt_;
    std::size_t cell_size_;

};
