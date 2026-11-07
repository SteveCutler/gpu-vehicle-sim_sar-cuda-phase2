#include <cuda_runtime.h>
#include "CUDA_simulation.hpp"

#include <iostream>

__global__ void CUDA_simulation::updateVehicles(float* vs_x, float* vs_y, float* vs_vx, float* vs_vy, float* vs_heading, float* vs_turnRate, const float* vs_goalx, const float* vs_goaly, std::size_t N, environment env, float dt){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N){
        return;
    }

    //create controller object
    Controller controller;

    //create dynamics updater
    Dynamics dynamics;

    //allocating variables outside hot loop
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

    //calculate new state
    newState = dynamics.step_update(vs, action, env, dt, step);

    //update old state
        vs_x[i] = newState.x;
        vs_y[i] = newState.y;
        vs_vx[i] = newState.vx;
        vs_vy[i] = newState.vy;
        vs_heading[i] = newState.heading;
        vs_turnRate[i] = newState.turnRate;

    return;
}

void CUDA_simulation::HtD_vehicles(){
    //create devices and allocate vehicle data
        const std::size_t floatBytes = N * sizeof(float);
     
        //creating devices for vehicle batch data, allocating memory and copying data over
        cudaMalloc(&this->vs_x, floatBytes);
        cudaMemcpy(this->vs_x, vehicles.m_x.data(), floatBytes, cudaMemcpyHostToDevice);
        
        cudaMalloc(&this->vs_y, floatBytes);
        cudaMemcpy(this->vs_y, vehicles.m_y.data(), floatBytes, cudaMemcpyHostToDevice);

        cudaMalloc(&this->vs_vx, floatBytes);
        cudaMemcpy(this->vs_vx, vehicles.m_vx.data(), floatBytes, cudaMemcpyHostToDevice);

        cudaMalloc(&this->vs_vy, floatBytes);
        cudaMemcpy(this->vs_vy, vehicles.m_vy.data(), floatBytes, cudaMemcpyHostToDevice);

        cudaMalloc(&this->vs_heading, floatBytes);
        cudaMemcpy(this->vs_heading, vehicles.m_heading.data(), floatBytes, cudaMemcpyHostToDevice);

        cudaMalloc(&this->vs_turnRate, floatBytes);
        cudaMemcpy(this->vs_turnRate, vehicles.m_turnRate.data(), floatBytes, cudaMemcpyHostToDevice);
        
        cudaMalloc(&this->vs_goalx, floatBytes);
        cudaMemcpy(this->vs_goalx, vehicles.m_goalx.data(), floatBytes, cudaMemcpyHostToDevice);

        cudaMalloc(&this->vs_goaly, floatBytes);
        cudaMemcpy(this->vs_goaly, vehicles.m_goaly.data(), floatBytes, cudaMemcpyHostToDevice);

        //check for allocation errors
        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

}

void CUDA_simulation::DtH_vehicles(){

    cudaMemcpy(vehicles.m_x.data(), vs_x, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_y.data(), vs_y, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_vx.data(), vs_vx, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_vy.data(), vs_vy, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_heading.data(), vs_heading, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_turnRate.data(), vs_turnRate, floatBytes, cudaMemcpyDeviceToHost);
}

void CUDA_simulation::init_simulation(
    std::size_t N,    
    vehicleBatch& vehicles,
    std::size_t particle_num,
    environment& env,
    std::size_t cell_count){
        
        //init cel count
        this->cell_count_ = cell_count;
        

        //handle fluid fields

        //handle particles

}

void CUDA_simulation::step_vehicles(){
        //threading setup
        std::size_t threads = 256;
        std::size_t blocks = (N + threads-1)/threads;
     
        //launch cuda kernel
        updateVehicles<<<blocks,threads>>>(this->vs_x, this->vs_y, this->vs_vx, this->vs_vy, this->vs_heading, this->vs_turnRate, this->vs_goalx, this->vs_goaly, this->v_count_, env, dt);
        
        //check for kernel launch errors
        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

}

void CUDA_simulation::runCudaSimulation(
    vehicleBatch& vehicles,
    environment& env,
    std::size_t N,
    float dt,
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


    //define loop

    //add step function
    std::size_t curr_step = 0;

    //run sim for steps amount of steps
    while(curr_step < steps){
        
        //advance vehicles
        HtD_vehicles();
        step_vehicles();
        DtH_vehicles()
    
        //scan for goal/kill particles
        target_found = vehicles.scan(newState, p_map, target);

        if(target_found){break;}

        //run planner every 50 steps
        if(curr_step % 50 == 0 && curr_step != 0){
            //choose waypoint based on particle drift predictions
            
            std::pair<float,float> newWP = planner.chooseWaypoint(vs, 
                p_map.particles_, 
                env, 
                dt, 
                tempMap,
                dynamics,
                controller,
                height);
            //update waypoint
            newState.goalx = newWP.first;
            newState.goaly = newWP.second;

            // std::cout << "Updating WP to: {" << newState.goalx << ", " << newState.goaly << "}" << std::endl;

            
        }
        
        //update old state
        vehicles.set(x, newState);


        //update target location
        driftDynamics.DriftUpdate(target, env.getDisturbance(target.x, target.y, 0.0f), dt);
    
        //update probability particles
        p_map.update(env, dt);
    
        //compute probability
        p_map.computeProb();
    
        env.update_fluid(dt);
    
    
        curr_step++;

    }

};


    
