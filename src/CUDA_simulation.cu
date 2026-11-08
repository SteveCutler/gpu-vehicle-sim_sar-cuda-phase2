#include <cuda_runtime.h>
#include "vehicle/CUDA_simulation.hpp"
#include <cuda/std/utility>
#include "vehicle/environment_CUDA.hpp"
#include "vehicle/controller_CUDA.hpp"
#include "vehicle/dynamics_CUDA.hpp"
#include "vehicle/vehicleTypes.hpp"

#include "vehicle/SearchPlanner_CUDA.hpp"
#include "vehicle/controller.hpp"
#include "vehicle/dynamics.hpp"



#include <iostream>

__device__ cuda::std::pair<float,float> vel_sample(    
    const float* u_prev,
    const float* v_prev,
    std::size_t width,
    std::size_t height,
    float x,
    float y){

        x = fmaxf(0.f, fminf(x, static_cast<float>(width-1.f)));
        y = fmaxf(0.f, fminf(y, static_cast<float>(height-1.f)));

        float u_vel = 0.f;
        float v_vel = 0.f;

        std::size_t x_floor = static_cast<std::size_t>(floorf(x)); 
        std::size_t x_high = (x_floor < width - 1) ? x_floor + 1 : x_floor;
        std::size_t y_floor = static_cast<std::size_t>(floorf(y));
        std::size_t y_high = (y_floor < height - 1) ? y_floor + 1 : y_floor;

        //finding values at corners
        std::size_t tl_index = x_floor+y_floor*width;
        std::size_t tr_index = x_high+y_floor*width;
        std::size_t bl_index = x_floor+y_high*width;
        std::size_t br_index = x_high+y_high*width;

        float tl_u = u_prev[tl_index];
        float tr_u = u_prev[tr_index];
        float bl_u = u_prev[bl_index];
        float br_u = u_prev[br_index];
        
        float tl_v = v_prev[tl_index];
        float tr_v = v_prev[tr_index];
        float bl_v = v_prev[bl_index];
        float br_v = v_prev[br_index];

        //distance from edges
        float dx = x - x_floor;
        float dy = y - y_floor;

        //weighting calculations
        float tl_weight = (1-dx)*(1-dy);
        float tr_weight = dx*(1-dy);
        float bl_weight = (1-dx)*dy;
        float br_weight = dx*dy;

        u_vel = (tl_u*tl_weight + tr_u*tr_weight + bl_u*bl_weight + br_u*br_weight);
        v_vel = (tl_v*tl_weight + tr_v*tr_weight + bl_v*bl_weight + br_v*br_weight);

        return{u_vel, v_vel};

}

__global__ void updateVehicles(float* vs_x, 
    float* vs_y, 
    float* vs_vx, 
    float* vs_vy, 
    float* vs_heading, 
    float* vs_turnRate, 
    const float* vs_goalx, 
    const float* vs_goaly, 
    const float* d_u_, 
    const float* d_v_, 
    const float* d_u_prev_, 
    const float* d_v_prev_, 
    std::size_t N, 
    float dt,
    std::size_t height,
    std::size_t width){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N){
        return;
    }

    //allocating variables outside hot loop
    ControllerCuda controller;
    DynamicsCuda dynamics;
    VehicleState vs;
    Action action;
    VehicleState newState;
    
    //create vehicleState object with thread vehicle
    vs = {
        vs_x[i],
        vs_y[i],
        vs_vx[i],
        vs_vy[i],
        vs_heading[i],
        vs_turnRate[i],
        vs_goalx[i],
        vs_goaly[i]
    };


    //pass to controller 
    action = controller.steer_controller(vs);

    cuda::std::pair<float, float> current = vel_sample(d_u_prev_, 
                                                    d_v_prev_, 
                                                    width, 
                                                    height, 
                                                    vs.x, 
                                                    vs.y);

    //calculate new state
    newState = dynamics.step_update(vs, action, current, dt);

    //update old state
        vs_x[i] = newState.x;
        vs_y[i] = newState.y;
        vs_vx[i] = newState.vx;
        vs_vy[i] = newState.vy;
        vs_heading[i] = newState.heading;
        vs_turnRate[i] = newState.turnRate;

    return;
}

void CUDA_simulation::HtD_vehicles(const vehicleBatch_CUDA& vehicles){
    //create devices and allocate vehicle data
        const std::size_t floatBytes = this->v_count_ * sizeof(float);
     
        //copy data from host to device
        cudaMemcpy(this->vs_x, vehicles.m_x.data(), floatBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->vs_y, vehicles.m_y.data(), floatBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->vs_vx, vehicles.m_vx.data(), floatBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->vs_vy, vehicles.m_vy.data(), floatBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->vs_heading, vehicles.m_heading.data(), floatBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->vs_turnRate, vehicles.m_turnRate.data(), floatBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->vs_goalx, vehicles.m_goalx.data(), floatBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->vs_goaly, vehicles.m_goaly.data(), floatBytes, cudaMemcpyHostToDevice);

        //check for allocation errors
        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

}

void CUDA_simulation::DtH_vehicles(vehicleBatch_CUDA& vehicles){
    const std::size_t floatBytes = v_count_ * sizeof(float);

    //copy vehicle data from device to host
    cudaMemcpy(vehicles.m_x.data(), vs_x, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_y.data(), vs_y, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_vx.data(), vs_vx, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_vy.data(), vs_vy, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_heading.data(), vs_heading, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_turnRate.data(), vs_turnRate, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_goalx.data(), vs_goalx, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_goaly.data(), vs_goaly, floatBytes, cudaMemcpyDeviceToHost);

        //check for allocation errors
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void CUDA_simulation::HtD_fluid(){
    //create devices and allocate vehicle data
        const std::size_t fluidBytes = this->env_.m_width * this->env_.m_height * sizeof(float);
     
        //copy data from host to device
        cudaMemcpy(this->d_u_, this->env_.fluid_grid.m_u_velocity.data(), fluidBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_v_, this->env_.fluid_grid.m_v_velocity.data(), fluidBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_u_prev_, this->env_.fluid_grid.m_u_velocity_prev.data(), fluidBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_v_prev_, this->env_.fluid_grid.m_v_velocity_prev.data(), fluidBytes, cudaMemcpyHostToDevice);

        //check for allocation errors
        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

}

void CUDA_simulation::DtH_fluid(){

    //copy vehicle data from device to host
    const std::size_t fluidBytes = this->env_.fluid_grid.m_width * this->env_.fluid_grid.m_height * sizeof(float);
    cudaMemcpy(this->env_.fluid_grid.m_u_velocity.data(), this->d_u_, fluidBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->env_.fluid_grid.m_v_velocity.data(), this->d_v_, fluidBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->env_.fluid_grid.m_u_velocity_prev.data(), this->d_u_prev_, fluidBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->env_.fluid_grid.m_v_velocity_prev.data(),this->d_v_prev_, fluidBytes, cudaMemcpyDeviceToHost);

        //check for allocation errors
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void CUDA_simulation::init_simulation( 
    vehicleBatch_CUDA& vehicles){
        
        //init cel count
       // this->cell_count_ = cell_count;

       
       //allocate devices for vehicle batch data, allocate here and copy over in HtDvehicles function
        const std::size_t floatBytes = this->v_count_ * sizeof(float);
        cudaMalloc(&this->vs_x, floatBytes);
        cudaMalloc(&this->vs_y, floatBytes);
        cudaMalloc(&this->vs_vx, floatBytes);
        cudaMalloc(&this->vs_vy, floatBytes);
        cudaMalloc(&this->vs_heading, floatBytes);
        cudaMalloc(&this->vs_turnRate, floatBytes);
        cudaMalloc(&this->vs_goalx, floatBytes);
        cudaMalloc(&this->vs_goaly, floatBytes);
        
        
        //allocate memory to devices for fluid fields
        const std::size_t fluidBytes = this->env_.m_width * this->env_.m_height * sizeof(float);
        cudaMalloc(&this->d_u_, fluidBytes);
        cudaMalloc(&this->d_v_, fluidBytes);
        cudaMalloc(&this->d_u_prev_, fluidBytes);
        cudaMalloc(&this->d_v_prev_, fluidBytes);


        //handle particles

}

void CUDA_simulation::step_vehicles(){
        //threading setup
        std::size_t threads = 256;
        std::size_t blocks = (this->v_count_ + threads-1)/threads;
     
        //launch cuda kernel
        updateVehicles<<<blocks,threads>>>(
            this->vs_x, 
            this->vs_y, 
            this->vs_vx, 
            this->vs_vy, 
            this->vs_heading, 
            this->vs_turnRate, 
            this->vs_goalx, 
            this->vs_goaly, 
            this->d_u_, 
            this->d_v_, 
            this->d_u_prev_, 
            this->d_v_prev_, 
            this->v_count_, 
            this->dt_, 
            this->env_.m_height, 
            this->env_.m_width);
        
        //check for kernel launch errors
        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

}

void CUDA_simulation::runCudaSimulation(
    vehicleBatch_CUDA& vehicles,
    std::size_t steps){


    //UPDATE FLOW
    //decide on action
    //calculate newState
    //check if target found -> if so break
    //if step%50 == 0 update waypoints -> include in state
    //update to new state

    //drift update
    //update particle map
    //compute probability
    //update fluid

    VehicleState vs;
    VehicleState newState;

    //add step function
    std::size_t curr_step = 0;

    //end condition
    bool target_found = false;

    //run sim for steps amount of steps
    while(curr_step < steps){
        
        //advance vehicles
        HtD_vehicles(vehicles);
        HtD_fluid();
        step_vehicles();
        DtH_vehicles(vehicles);
    
        //scan for goal/kill particles
        for(int v = 0; v < this->v_count_; v++){
            vs = vehicles.load(v);

            target_found = vehicles.scan(vs, this->p_map_, this->target_);
    
            if (target_found) {
                vehicles.set(v, vs);
                break;
            }
    
            //run planner every 50 steps
            if(curr_step % 50 == 0 && curr_step != 0){
                //choose waypoint based on particle drift predictions
                
                std::pair<float,float> newWP = this->planner_.chooseWaypoint(
                    vs, 
                    this->p_map_.particles_, 
                    this->env_, 
                    this->dt_, 
                    this->temp_map_,
                    this->dynamics,
                    this->controller,
                    this->env_.m_height);
                //update waypoint
                vs.goalx = newWP.first;
                vs.goaly = newWP.second;
    
                // std::cout << "Updating WP to: {" << newState.goalx << ", " << newState.goaly << "}" << std::endl;
    
                
            }
            //update old state
            vehicles.set(v, vs);
        }

        if (target_found) {
            break;
        }
        


        //update target location
        this->p_map_.DD_.DriftUpdate(this->target_, env_.getDisturbance(this->target_.x, this->target_.y, 0.0f), this->dt_);
    
        //update probability particles
        this->p_map_.update(this->env_, this->dt_);
    
        //compute probability
        this->p_map_.computeProb();
    
        this->env_.update_fluid(this->dt_);
    
    
        curr_step++;

    }

};

CUDA_simulation::~CUDA_simulation(){
        //free allocated memory
    float* allocations[] = {
        vs_x, 
        vs_y, 
        vs_vx, 
        vs_vy, 
        vs_heading, 
        vs_turnRate, 
        vs_goalx, 
        vs_goaly,
        d_u_,
        d_v_,
        d_u_prev_,
        d_v_prev_        
    };

    for (float* ptr : allocations){
        cudaFree(ptr);
    }
}


    
