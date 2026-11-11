#include <cuda_runtime.h>
#include "vehicle/CUDA_simulation.hpp"
#include <cuda/std/utility>
#include "vehicle/environment.hpp"
#include "vehicle/controller_CUDA.hpp"
#include "vehicle/dynamics_CUDA.hpp"
#include "vehicle/vehicleTypes.hpp"
#include <vector>
#include "vehicle/SearchPlanner.hpp"
#include "vehicle/constants.hpp"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>

//cuda error check function, wrapping all mallocs and transfers with it
inline void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}
#define CUDA_CHECK(call) cudaCheck(call)

//velocity field sample device helper function
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


__device__ bool scan_helper(
    const float px, 
    const float py, 
    const float vs_x, 
    const float vs_y, 
    const float sensor_radius){

    
    float dist = (px - vs_x) * (px - vs_x) + (py - vs_y) * (py - vs_y);
    if(dist < (sensor_radius*2*sensor_radius*2)){
        return true;
    }
    return false;
}

__global__ void particle_scan_kernel(
    const PredictionParticle* d_particles_, 
    unsigned char* d_remove_particles_, 
    const float* vs_x, 
    const float* vs_y, 
    float sensor_radius_, 
    std::size_t v_count_,
    std::size_t p_count_,
    std::size_t height
    ){

        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        //bounds check
        if(idx >= p_count_){
            return;
        }

        //every thread takes a particle
        float p_x_pos = d_particles_[idx].pos.x;
        float p_y_pos = height - d_particles_[idx].pos.y;
        //detection sentinel
        unsigned char detected = 0;
        

        //loop through all vehicles to determine if particle is within any sensor radius
        for(int v = 0; v < v_count_; v++){
            if(scan_helper(p_x_pos, p_y_pos, vs_x[v], vs_y[v], sensor_radius_)){
                detected = 1;
                break;
            }
        }
        //write out detection flag
        d_remove_particles_[idx] = detected;

};

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

__global__ void p_horizons(
    const PredictionParticle* particles, 
    PredictionParticle* forecasts,
    const float* d_u_prev,
    const float* d_v_prev,
    std::size_t steps,
    std::size_t p_count,
    std::size_t height,
    std::size_t width,
    float dt
){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(idx >= p_count){
        return;
    }
    
    //write the first position to forecasts as step 0
    PredictionParticle particle = particles[idx];
    forecasts[idx] = particle; 

    //advance particle through all steps and write to return array

    for(std::size_t s = 1; s <= steps; s++){
        
        
        //use vel sample device function to sample curent at pos
        cuda::std::pair<float, float> current = vel_sample(d_u_prev, 
                                                d_v_prev, 
                                                width, 
                                                height, 
                                                particle.pos.x,
                                                particle.pos.y);

                                                //advect particles using current sample
        particle.pos.x += current.first * dt;
        particle.pos.y += current.second * dt;

        forecasts[idx + (s * p_count)] = particle;
    }

}

__global__ void occupancy_kernel(
    unsigned int* maps, 
    PredictionParticle* forecasts, 
    std::size_t steps, 
    std::size_t p_count, 
    std::size_t cell_size,
    std::size_t cell_count,
    std::size_t height,
    std::size_t width){

    //one thread for every particle
    std::size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx >= p_count){
        return;
    }

    //calculate starting indices
    std::size_t s1 = p_count * (steps/2);
    std::size_t s2 = p_count * (steps);

    //grab correct particles
    PredictionParticle p1 = forecasts[s1 + idx];
    PredictionParticle p2 = forecasts[s2 + idx];

    // Each forecast sample can independently leave or re-enter the domain.
    std::size_t rows = height / cell_size;
    std::size_t cols = width / cell_size;

    if(p1.pos.x >= 0.f && p1.pos.x < width &&
       p1.pos.y >= 0.f && p1.pos.y < height){
        std::size_t col = static_cast<std::size_t>(floorf(p1.pos.x / cell_size));
        std::size_t row = static_cast<std::size_t>(floorf(p1.pos.y / cell_size));
        if(col < cols && row < rows){
            atomicAdd(&maps[row * cols + col], 1u);
        }
    }
    if(p2.pos.x >= 0.f && p2.pos.x < width &&
       p2.pos.y >= 0.f && p2.pos.y < height){
        std::size_t col = static_cast<std::size_t>(floorf(p2.pos.x / cell_size));
        std::size_t row = static_cast<std::size_t>(floorf(p2.pos.y / cell_size));
        if(col < cols && row < rows){
            atomicAdd(&maps[cell_count + row * cols + col], 1u);
        }
    }

};

__device__ Action controller_update(
    const VehicleState& vs
){

    constexpr float pi = 3.14159265359f;

    //use arctangent to calculate desired heading angle
    const float desiredHeading = atan2f(vs.goaly - vs.y, vs.goalx - vs.x);

    //calculate shortest angle corrective between current heading and desired heading
    const float headingError = remainderf(desiredHeading - vs.heading, 2.0f * pi);

    //PD controls
    constexpr float kp = constants::kp;
    constexpr float kd = constants::kd;
    constexpr float maxTorque = constants::maxTorque;

     Action action{};
    action.torque = fminf(maxTorque, fmaxf(
        -maxTorque, kp * headingError - kd * vs.turnRate
    ));

    //create dynamic thrust
    constexpr float distGain = constants::distGain;
    constexpr float speedGain = constants::speedGain;
    constexpr float maxSpeed = constants::maxSpeed;
    constexpr float maxThrust = constants::maxThrust;

    float distx = vs.x - vs.goalx;
    float disty = vs.y - vs.goaly;
    float dist = sqrtf(distx*distx + disty*disty);

    //measure if pointing towards goal
    float alignment = fmaxf(0.0f, cosf(headingError));

    //clamp speed at max and multiply by direction pointing
    float desiredSpeed = fminf(maxSpeed, dist * distGain) * alignment;

    //measure actual speed
    float actualSpeed = vs.vx * cosf(vs.heading) 
                        + vs.vy * sinf(vs.heading);

    //discrepency between desired and actual
    float speedError = desiredSpeed - actualSpeed;

    action.thrust = fminf(maxThrust,
        fmaxf(-maxThrust, speedError * speedGain)
    );

    return action;
}

__device__ void dynamics_update(
    VehicleState& vs,
    const Action& action,
    const float dt,
    const float* d_u_prev,
    const float* d_v_prev,
    const std::size_t height,
    const std::size_t width
){

    //initial params
    constexpr float mass = constants::mass;
    constexpr float momentOfInertia = constants::momentOfInertia;
    constexpr float linearDrag = constants::linearDrag;
    constexpr float turnDrag = constants::turnDrag;
    constexpr float pi = 3.14159265359f;

    //retrieve ocean disturbance at this position
    cuda::std::pair<float, float> current = vel_sample(d_u_prev, 
                                                    d_v_prev, 
                                                    width, 
                                                    height, 
                                                    vs.x, 
                                                    vs.y);


    //Forces in x y coords
    const float thrustX = action.thrust * cosf(vs.heading);
    const float thrustY = action.thrust * sinf(vs.heading);

    //current is 0 for this first implementation
    const float relativeVx = vs.vx - current.first;
    const float relativeVy = vs.vy - current.second;

    //accel x and y
    const float ax = (thrustX - linearDrag * relativeVx) / mass;
    const float ay = (thrustY - linearDrag * relativeVy) / mass;

    //angular accel
    const float angularAcceleration = (action.torque - turnDrag * vs.turnRate) / momentOfInertia;

    //copy old state to preserve the things we aren't changing
    //VehicleState newVs = vs;

    vs.vx = vs.vx + ax * dt;
    vs.vy = vs.vy + ay * dt;
    vs.x = vs.x + vs.vx * dt;
    vs.y = vs.y + vs.vy * dt;

    vs.turnRate = vs.turnRate + angularAcceleration * dt;
    vs.heading = remainderf(vs.heading + vs.turnRate * dt, 2.f * pi);

}


__global__ void path_kernel(
    //vehicle vars
    float* vs_x,
    float* vs_y, 
    float* vs_vx, 
    float* vs_vy, 
    float* vs_heading, 
    float* vs_turnRate, 
    float* vs_goalx, 
    float* vs_goaly, 
    //fluid vars
    const float* d_u_prev_, 
    const float* d_v_prev_, 
    //particle vars
    const PredictionParticle* p_forecasts,
    unsigned char* hit_map,
    std::size_t p_count,
    //candidate waypoints
    const float* waypoints_x,
    const float* waypoints_y,

    const std::size_t v_count, 
    const float dt,
    const float sensor_rad,
    const std::size_t wp_num,
    const std::size_t height,
    const std::size_t width,
    const std::size_t steps


){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx >= p_count){
        return;
    }
    VehicleState vs;

    //for every vehicle, 
    for(int v = 0; v < v_count; v++){
        
        //for its waypoints, loop through every step, simulating the vehicle advancing towards the waypoint
        for(int wp = 0; wp < wp_num; wp++){
            //load vehicle original state to reset starting state for each wp run
            vs = {
                vs_x[v],
                vs_y[v],
                vs_vx[v],
                vs_vy[v],
                vs_heading[v],
                vs_turnRate[v],
                vs_goalx[v],
                vs_goaly[v]
            };
            //first run keep the same but then update to new candidate waypoints after
            if(wp != 0){
                vs.goalx = waypoints_x[wp-1];
                vs.goaly = waypoints_y[wp-1];
            }

            //calculate current offset for hitmap
            std::size_t offset = (wp * p_count) + (v * wp_num * p_count);

            //current step set to 0
            hit_map[offset + idx] = 0;

            //for each step calculate new vehicle pos given this waypoint
            //loop starts at 1 to exclude current step
            for(std::size_t s = 1; s <= steps; s++){
                //pass vehicle state by reference, this updates to new state
                dynamics_update(vs,
                    controller_update(vs),
                    dt,
                    d_u_prev_,
                    d_v_prev_,
                    height,
                    width);

                //for each particle check to see if it is in the sensor radius of the new vehicle state
            
                const PredictionParticle particle = p_forecasts[p_count * s + idx];
                if(scan_helper(
                    particle.pos.x,
                    static_cast<float>(height) - particle.pos.y,
                    vs.x,
                    vs.y,
                    sensor_rad
                )){
                    //particle collision has been found
                    hit_map[idx + offset] = 1;
                    break;
                }
                

            }

        }

    }
}

__global__ void score_paths_kernel(
    const unsigned char* hit_num,
    const std::size_t wp_num,
    const std::size_t v_count,
    const std::size_t p_count,
    unsigned int* best_wps,
    const std::size_t smem_num
){

    extern __shared__ unsigned int sMem[];

    
    //first loop just fills in the smem, other loops compare
    int idx = threadIdx.x;
    
    //each block has 256 threads and is used to calculate all rollouts for one vehicle
    //
    
    if(blockIdx.x >= v_count){
        return;
    }
    unsigned int hit_tally = 0;
    unsigned int best_wp = 0;
    
    int warp_num = idx / 32;
    int lane_num = idx % 32;
    
    int v_num = blockIdx.x;
    unsigned int FULL_MASK = 0xFFFFFFFF;
    std::size_t hit_offset;
    
    //loop over all wp for current vehicle
    for(int wp = 0; wp < wp_num; wp++){
        unsigned int local_hits = 0;
        //calculate hit_offset, vehicle * waypoints * particle counts to access correct hit_num data
        hit_offset = (v_num * wp_num + wp) * p_count;

        //within current wp, threads only go up to p_coun

        for(std::size_t tid = idx; tid < p_count; tid += blockDim.x){
    
    
            //load hit data
            local_hits += hit_num[tid+hit_offset];
            
        }        
        
        //add thread local tallys to sMem slots based on warp num
        
        //reduce warp level tallys
        int offset = 1;
        while(offset < 32){
            local_hits += __shfl_down_sync(FULL_MASK, local_hits, offset);
            offset*=2;
        }
        
        if(lane_num == 0){
            sMem[warp_num] = local_hits;
        }
        __syncthreads();
        
        //thread 1 reduces sMemresults and stores in register value
        if(warp_num == 0){
            //if lane in bounds for smem, load value, else get 0
            unsigned int warp_total = lane_num < smem_num ? sMem[lane_num] : 0;
            
            offset = 1;
            while(offset < 32){
                warp_total += __shfl_down_sync(FULL_MASK, warp_total, offset);
                offset *=2;
            }
            
            if(lane_num == 0){
                if(warp_total > hit_tally){
                    best_wp = wp;
                    hit_tally = warp_total;
                }
            }
        }
        __syncthreads();
        
    }

    //afterwards tid 0 writes out best_wp to 
    if(idx == 0){

        best_wps[v_num] = best_wp;
    }

}

void CUDA_simulation::init_simulation( 
    vehicleBatch& vehicles){
        
        //init cel count
       // this->cell_count_ = cell_count;
       if(this->v_count_ != 0){
           this->sensor_radius_ = vehicles.m_sensor_rad[0];
       }

       //calculate space requirements for the different devices
       const std::size_t floatBytes = this->v_count_ * sizeof(float);
       const std::size_t fluidBytes = this->env_.m_width * this->env_.m_height * sizeof(float);
       //particles are AoS layout
       const std::size_t pBytes = this->p_map_.particles_.size() * sizeof(PredictionParticle);
       const std::size_t flagBytes = this->p_map_.particles_.size() * sizeof(unsigned char);

        std::size_t steps = this->planner_.horizon_ + 1;
        std::size_t p_count = this->p_map_.particles_.size();
        std::size_t p_horizons_Bytes = p_count * steps * sizeof(PredictionParticle);

        //hit map bytes
        std::size_t hit_bytes = this->v_count_ * this->candidate_waypoints_ * p_count * sizeof(unsigned char);

       
       //allocate devices for vehicle batch data, allocate here and copy over in HtDvehicles function
        CUDA_CHECK(cudaMalloc(&this->vs_x, floatBytes));
        CUDA_CHECK(cudaMalloc(&this->vs_y, floatBytes));
        CUDA_CHECK(cudaMalloc(&this->vs_vx, floatBytes));
        CUDA_CHECK(cudaMalloc(&this->vs_vy, floatBytes));
        CUDA_CHECK(cudaMalloc(&this->vs_heading, floatBytes));
        CUDA_CHECK(cudaMalloc(&this->vs_turnRate, floatBytes));
        CUDA_CHECK(cudaMalloc(&this->vs_goalx, floatBytes));
        CUDA_CHECK(cudaMalloc(&this->vs_goaly, floatBytes));
        
        
        //allocate memory to devices for fluid fields
        CUDA_CHECK(cudaMalloc(&this->d_noise_, fluidBytes));

        // density
        CUDA_CHECK(cudaMalloc(&this->d_density_r_, fluidBytes));
        CUDA_CHECK(cudaMalloc(&this->d_density_g_, fluidBytes));
        CUDA_CHECK(cudaMalloc(&this->d_density_b_, fluidBytes));
        CUDA_CHECK(cudaMalloc(&this->d_density_r_prev_, fluidBytes));
        CUDA_CHECK(cudaMalloc(&this->d_density_g_prev_, fluidBytes));
        CUDA_CHECK(cudaMalloc(&this->d_density_b_prev_, fluidBytes));
        
        
        // Projection
        CUDA_CHECK(cudaMalloc(&this->d_divergence_, fluidBytes));
        CUDA_CHECK(cudaMalloc(&this->d_pressure_, fluidBytes));
        CUDA_CHECK(cudaMalloc(&this->d_pressure_prev_, fluidBytes));
        CUDA_CHECK(cudaMalloc(&this->d_diffusion_scratch_, fluidBytes));

        CUDA_CHECK(cudaMalloc(&this->d_u_, fluidBytes));
        CUDA_CHECK(cudaMalloc(&this->d_v_, fluidBytes));
        CUDA_CHECK(cudaMalloc(&this->d_u_prev_, fluidBytes));
        CUDA_CHECK(cudaMalloc(&this->d_v_prev_, fluidBytes));


        // Initialize all fluid fields once, before the first kernel reads them.
        HtD_fluid();

        //allocate memory to particle device
        CUDA_CHECK(cudaMalloc(&this->d_particles_, pBytes));
        CUDA_CHECK(cudaMalloc(&this->d_remove_particles_, flagBytes));
        CUDA_CHECK(cudaMalloc(&this->d_p_horizons, p_horizons_Bytes));
        
        //allocate space for particle horizons
        this->horizons_.resize(p_count * steps );
        //allocate space for candidate waypoint calculations
        this->planner_.wayPoints.resize(this->candidate_waypoints_);
        CUDA_CHECK(cudaMalloc(&this->d_waypoints_x, (this->candidate_waypoints_-1)*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&this->d_waypoints_y, (this->candidate_waypoints_-1)*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&this->d_best_wp, (this->v_count_ * sizeof(unsigned int))));

        //occupancy maps -> -1 because the first waypoint candidate is the current one
        this->cell_count_ = (env_.m_width / cell_size_) * (env_.m_height / cell_size_);
        CUDA_CHECK(cudaMalloc(&this->d_occupancy_maps_, (cell_count_ * (this->candidate_waypoints_ - 1) * sizeof(unsigned int))));
        this->occupancy_maps_.resize(cell_count_ * (this->candidate_waypoints_ - 1));

        //calc number of cells
        //allocate space for hit maps for path calculations
        CUDA_CHECK(cudaMalloc(&this->d_hit_map, hit_bytes));


}

void CUDA_simulation::HtD_vehicles(const vehicleBatch& vehicles){
    //create devices and allocate vehicle data
        const std::size_t floatBytes = this->v_count_ * sizeof(float);
     
        //copy data from host to device
        CUDA_CHECK(cudaMemcpy(this->vs_x, vehicles.m_x.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(this->vs_y, vehicles.m_y.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(this->vs_vx, vehicles.m_vx.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(this->vs_vy, vehicles.m_vy.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(this->vs_heading, vehicles.m_heading.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(this->vs_turnRate, vehicles.m_turnRate.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(this->vs_goalx, vehicles.m_goalx.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(this->vs_goaly, vehicles.m_goaly.data(), floatBytes, cudaMemcpyHostToDevice));

        //check for allocation errors
        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

}

void CUDA_simulation::DtH_vehicles(vehicleBatch& vehicles){
    const std::size_t floatBytes = v_count_ * sizeof(float);

    //copy vehicle data from device to host
    CUDA_CHECK(cudaMemcpy(vehicles.m_x.data(), vs_x, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vehicles.m_y.data(), vs_y, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vehicles.m_vx.data(), vs_vx, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vehicles.m_vy.data(), vs_vy, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vehicles.m_heading.data(), vs_heading, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vehicles.m_turnRate.data(), vs_turnRate, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vehicles.m_goalx.data(), vs_goalx, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vehicles.m_goaly.data(), vs_goaly, floatBytes, cudaMemcpyDeviceToHost));

        //check for allocation errors
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void CUDA_simulation::particle_scan(){
    
    //calc particle num
    const std::size_t p_count = p_map_.particles_.size();
    auto& particles = p_map_.particles_;

    if (p_count == 0) {
        return;
    }

    std::vector<unsigned char> remove_particles(p_count);

    //transfer particle data to GPU
    CUDA_CHECK(cudaMemcpy(this->d_particles_, particles.data(), p_count*sizeof(PredictionParticle), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (p_count + threads - 1) / threads;

    //launch GPU kernel -> send particle device, vehicle device, return list of particles to cull
    particle_scan_kernel<<<blocks, threads>>>(
        this->d_particles_, 
        this->d_remove_particles_, 
        this->vs_x, 
        this->vs_y, 
        this->sensor_radius_, 
        this->v_count_, 
        p_count,
        static_cast<float>(env_.m_height)
        );
    CUDA_CHECK(cudaGetLastError());
    
    //transfer back hits and remove them from p_map.particles_
    CUDA_CHECK(cudaMemcpy(remove_particles.data(), d_remove_particles_, sizeof(unsigned char) * p_count, cudaMemcpyDeviceToHost));

    //remove collision particles
    int write = 0;
    for(int read = 0; read < particles.size(); read++){
        if(remove_particles[read] == 0){
            if(write != read){
                particles[write] = particles[read];
            }
            write++;
        }

    }

    particles.resize(write);
    //later we can optimize by adding a binned cell based check


}
void CUDA_simulation::HtD_fluid(){
    const std::size_t fluidBytes = env_.m_width * env_.m_height * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_u_, env_.fluid_grid.m_u_velocity.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_, env_.fluid_grid.m_v_velocity.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_prev_, env_.fluid_grid.m_u_velocity_prev.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_prev_, env_.fluid_grid.m_v_velocity_prev.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_noise_, env_.fluid_grid.m_noise.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_density_r_, env_.fluid_grid.m_density_r.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_density_g_, env_.fluid_grid.m_density_g.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_density_b_, env_.fluid_grid.m_density_b.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_density_r_prev_, env_.fluid_grid.m_density_r_prev.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_density_g_prev_, env_.fluid_grid.m_density_g_prev.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_density_b_prev_, env_.fluid_grid.m_density_b_prev.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_divergence_, env_.fluid_grid.m_divergence.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pressure_, env_.fluid_grid.m_pressure.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pressure_prev_, env_.fluid_grid.m_pressure_prev.data(), fluidBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_diffusion_scratch_, env_.fluid_grid.m_diffusion_scratch.data(), fluidBytes, cudaMemcpyHostToDevice));
}

void CUDA_simulation::DtH_fluid(){
    const std::size_t fluidBytes = env_.m_width * env_.m_height * sizeof(float);
    CUDA_CHECK(cudaMemcpy(env_.fluid_grid.m_u_velocity.data(), d_u_, fluidBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(env_.fluid_grid.m_v_velocity.data(), d_v_, fluidBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(env_.fluid_grid.m_u_velocity_prev.data(), d_u_prev_, fluidBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(env_.fluid_grid.m_v_velocity_prev.data(), d_v_prev_, fluidBytes, cudaMemcpyDeviceToHost));
}

void CUDA_simulation::step_vehicles(){
        if(v_count_ == 0) return;
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

void CUDA_simulation::calc_p_horizon(std::size_t steps){
    
    //need horizon device holder
    std::size_t p_count = p_map_.particles_.size();
    if(p_count == 0) return;

    //fluid devices still contain appropriate data

    std::size_t threads = 128;
    std::size_t blocks = (p_count + threads - 1)/threads;

    //update particle data for GPU
    CUDA_CHECK(cudaMemcpy(
    this->d_particles_,
    this->p_map_.particles_.data(),
    p_count * sizeof(PredictionParticle),
    cudaMemcpyHostToDevice));

    //kernel laucnh

    p_horizons<<<blocks, threads>>>(
        this->d_particles_, 
        this->d_p_horizons,
        this->d_u_prev_,
        this->d_v_prev_,
        steps,
        p_count,
        this->env_.m_height,
        this->env_.m_width,
        this->dt_);
    CUDA_CHECK(cudaGetLastError());

        //copy data back over, save to member variable

        // cudaMemcpy(this->horizons_.data(), 
        // this->d_p_horizons, 
        // p_count * steps * sizeof(PredictionParticle), 
        // cudaMemcpyDeviceToHost);


    

}

void CUDA_simulation::calc_occ_maps(
    std::size_t steps, 
    std::size_t p_count){

        //create two empty occupancy grids to pass in
        std::fill(occupancy_maps_.begin(), occupancy_maps_.end(), 0u);

        std::size_t map_bytes = 
            this->cell_count_ * (this->candidate_waypoints_ - 1) * sizeof(unsigned int);
        CUDA_CHECK(cudaMemset(this->d_occupancy_maps_, 0, map_bytes));


        int threads = 256;
        int blocks = (p_count + threads - 1)/threads;
        
        if(p_count > 0){

            // launch kernel -> compute occupancy at key horizons to determine candidate waypoints
           occupancy_kernel<<<blocks, threads>>>(
                this->d_occupancy_maps_,
                this->d_p_horizons, 
                steps,
                p_count, 
                this->cell_size_, 
                this->cell_count_,
                this->env_.m_height,
                this->env_.m_width);
    CUDA_CHECK(cudaGetLastError());
            
            //transfer waypoints back to host
            CUDA_CHECK(cudaMemcpy(this->occupancy_maps_.data(),
            this->d_occupancy_maps_,
            map_bytes, 
            cudaMemcpyDeviceToHost
            ));

        }

    }

std::pair<float,float> CUDA_simulation::convert_cell(std::size_t cell){
    
    std::size_t cols = this->env_.m_width / this->cell_size_;

    std::size_t col = cell % cols;
    std::size_t row = cell / cols;

    // Centre of the selected cell.
    float goalx = (static_cast<float>(col) + 0.5f) * this->cell_size_;
    float goaly = this->env_.m_height - (static_cast<float>(row) + 0.5f) * this->cell_size_;
    return {goalx, goaly};
};


void CUDA_simulation::find_waypoints(){

    for(int wp = 0; wp <= 1; wp++){

        unsigned int max = 0;
        std::size_t max_cell = 0;

        //loop through section of occupancy map
        for(int cell = 0 + (wp*this->cell_count_); cell < ((wp+1) * this->cell_count_); cell++){
            if(this->occupancy_maps_[cell] > max){
                max_cell = cell - (wp*this->cell_count_);
                max = this->occupancy_maps_[cell];
            }

        }

        this->planner_.wayPoints[wp+1] = convert_cell(max_cell);

    }

}

void CUDA_simulation::calc_paths(
    vehicleBatch& vehicles, 
    std::size_t steps, 
    std::size_t p_count
){
    if(p_count == 0 || v_count_ == 0) return;

    // CPU goal checks may have changed candidate zero after the last download.
    HtD_vehicles(vehicles);

    //reset hit maps to 0
    std::size_t hit_bytes = this->v_count_ * this->candidate_waypoints_ * p_count * sizeof(unsigned char);
    CUDA_CHECK(cudaMemset(d_hit_map, 0, hit_bytes));

    //waypoint component vectors (-1 because the first is the current, should resize original array to simplify this math)
    std::vector<float> wp_x(this->candidate_waypoints_-1);
    std::vector<float> wp_y(this->candidate_waypoints_-1);

    //move candidate waypoints into devices
    for(int i = 1; i < this->candidate_waypoints_; i++){
        wp_x[i-1] = this->planner_.wayPoints[i].first;
        wp_y[i-1] = this->planner_.wayPoints[i].second;
    }

    CUDA_CHECK(cudaMemcpy(this->d_waypoints_x, wp_x.data(), wp_x.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(this->d_waypoints_y, wp_y.data(), wp_y.size()*sizeof(float), cudaMemcpyHostToDevice));

    std::size_t threads = 256;
    std::size_t blocks = (p_count + threads - 1) / threads;


    //current goals are being passed in as part of vehicle state
    //alternative waypoints are being passed in in the d_way


    path_kernel<<<blocks, threads>>>(
        this->vs_x, 
        this->vs_y,
        this->vs_vx,
        this->vs_vy,
        this->vs_heading,
        this->vs_turnRate,
        this->vs_goalx,
        this->vs_goaly,
        this->d_u_prev_,
        this->d_v_prev_,
        this->d_p_horizons,
        this->d_hit_map,
        p_count,
        this->d_waypoints_x,
        this->d_waypoints_y,
        this->v_count_,
        this->dt_,
        this->sensor_radius_,
        this->candidate_waypoints_,
        this->env_.m_height,
        this->env_.m_width,
        steps
        );
    CUDA_CHECK(cudaGetLastError());

        //particle hits are in d_hitmap

}

void CUDA_simulation::score_paths(vehicleBatch& vehicles){
    if(v_count_ == 0 || p_map_.particles_.empty()) return;

    //one block for every vehicle, loops over all waypoint rollouts and calculates hit totals
    std::size_t p_count = p_map_.particles_.size();
    //hit total return vectors
    std::size_t paths = this->v_count_;
    std::vector<unsigned int> best_wps(paths);



    int threads = 256;
    int blocks = this->v_count_;

    //smem is threads / warp size
    std::size_t warp_count = threads / 32;
    //8 unsigned-int warp totals: 32 bytes per block
    std::size_t smem_bytes = warp_count * sizeof(unsigned int);

    //pass in d_hit_num
    score_paths_kernel<<<blocks, threads, smem_bytes>>>(
        this->d_hit_map, 
        this->candidate_waypoints_,
        this->v_count_,
        p_count,
        this->d_best_wp,
        warp_count

    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(best_wps.data(),
    d_best_wp, 
    paths * sizeof(unsigned int), 
    cudaMemcpyDeviceToHost));

    //loop over results and assign waypoints to the vehicles
    for(std::size_t v = 0; v < this->v_count_; v++){
        const unsigned int winner = best_wps[v];

        if (winner != 0) {
            VehicleState state = vehicles.load(v);
            state.goalx = planner_.wayPoints[winner].first;
            state.goaly = planner_.wayPoints[winner].second;
            vehicles.set(v, state);
        }
    }

};


std::size_t CUDA_simulation::runCudaSimulation(
    vehicleBatch& vehicles,
    std::size_t steps){

    VehicleState vs;
    VehicleState newState;

    //add step function
    std::size_t curr_step = 0;

    //end condition
    bool target_found = false;

    //run sim for steps amount of steps
    while(curr_step < steps){
        
        // Fluid fields stay on the GPU between steps.
        HtD_vehicles(vehicles);
        //advance vehicles
        step_vehicles();

        //copy vehicle data back for comparison
        DtH_vehicles(vehicles);

        particle_scan();

            //scan for goal/target
            for(int v = 0; v < this->v_count_; v++){
                vs = vehicles.load(v);
                
                //CPU checks if target or goal are inside sensor radius
                target_found = vehicles.targetCheck(vs, this->target_, env_.m_height);
                
                if (target_found) {
                    vehicles.set(v, vs);
                    break;
                }
        
                vehicles.goalCheck(vs, this->p_map_);

                vehicles.set(v, vs);
        
            }

            if (target_found) {
                ++curr_step;
                break;
            }


        //once every 50 steps create particle forecast to be used for waypoint calculation

        if(curr_step % 50 == 0 && curr_step != 0 && !p_map_.particles_.empty()){

            //Waypoint Recalculation -> break into 4 seperate operations 

            //particle horizon forecast
            calc_p_horizon(this->planner_.horizon_);

            //build midpoint/endpoint occupancy maps and return candidate paths


            calc_occ_maps( 
                this->planner_.horizon_, 
                this->p_map_.particles_.size()
            );

            find_waypoints();

            calc_paths(
                vehicles,
                this->planner_.horizon_,
                this->p_map_.particles_.size()
            );

            score_paths(vehicles);

        }
        
        

        // STILL NEED TO BE PORTED TO CUDA

        //update target location
        this->p_map_.DD_.DriftUpdate(this->target_, env_.getDisturbance(this->target_.x, this->target_.y, 0.0f), this->dt_);
    
        //update probability particles
        this->p_map_.update(this->env_, this->dt_);
    
        //compute probability
        this->p_map_.computeProb();
    

        //CUDA fluid update
        step_fluid(this->dt_);
    
    
        curr_step++;

    }

    return curr_step;
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
        d_v_prev_,
        d_noise_,
        d_density_r_,
        d_density_g_,
        d_density_b_,
        d_density_r_prev_,
        d_density_g_prev_,
        d_density_b_prev_,
        d_divergence_,
        d_pressure_,
        d_pressure_prev_,
        d_diffusion_scratch_,
    };
    //free float array pointers
    for (float* ptr : allocations){
        cudaFree(ptr);
    }
    //free other allocations
    cudaFree(d_particles_);
    cudaFree(d_remove_particles_);    
    cudaFree(d_p_horizons);
    cudaFree(d_hit_map);
    cudaFree(d_occupancy_maps_);
    cudaFree(d_waypoints_x);
    cudaFree(d_waypoints_y);
    cudaFree(d_best_wp);

    
}
