
#include "vehicle/ProbabilityMap_CUDA.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include "vehicle/DriftDynamics.hpp"
#include <cuda_runtime.h>

ProbabilityMap::ProbabilityMap(DriftDynamics& DD, std::size_t width, std::size_t height, std::size_t num_p, float est_x, float est_y, float p_rad, std::size_t cell_size):
width_(width),
height_(height),
num_p_(num_p),
cell_size_(cell_size),
map_((width/cell_size_) * (height/cell_size_), 0.f),
DD_(DD),
rng_(25),
p_rad_(p_rad),
origin_x_(est_x),
origin_y_(est_y),
target_cell(0)
{
   

add_Particles();


};

void ProbabilityMap::update(const environment_CUDA& env, float dt){

   // add_Particles();

    //update probability particle drift

    //scan for dead particles, if alive update dynamics
    for(auto it = particles_.begin(); it != particles_.end();){
        it->life_ -= dt;
        if( it->life_ <0.f){
            it = particles_.erase(it);
        }else{
             DD_.DriftUpdate(it->pos, env.getDisturbance(it->pos.x, it->pos.y, 0.f), dt);
             it++;
        }
    }


};
void ProbabilityMap::updateGoal(VehicleState& vs){
    std::size_t cols = width_ / cell_size_;

    std::size_t col = target_cell % cols;
    std::size_t row = target_cell / cols;

    // Centre of the selected cell.
    vs.goalx = (static_cast<float>(col) + 0.5f) * cell_size_;
    vs.goaly = height_ - (static_cast<float>(row) + 0.5f) * cell_size_;
    return;
};


std::pair<float,float> ProbabilityMap::convertCell(std::size_t cell){
        std::size_t cols = width_ / cell_size_;

    std::size_t col = cell % cols;
    std::size_t row = cell / cols;
    float cellx = (static_cast<float>(col) + 0.5f) * cell_size_;
    float celly = height_ - (static_cast<float>(row) + 0.5f) * cell_size_;

    return {cellx, celly};

}

void ProbabilityMap::particleScan(VehicleState& vs){

    for(auto it = particles_.begin(); it < particles_.end();){
        float x_pos = it->pos.x;
        float y_pos = it->pos.y;

        float x_dist = abs(x_pos - vs.x);
        float y_dist = abs(y_pos - (height_ - vs.y));

        
        if(x_dist*x_dist + y_dist*y_dist < (vs.sensor_rad*2)*(2* vs.sensor_rad)){
            it = particles_.erase(it);
        }else{

            it++;
        }
    }
    //std::cout << "particle scan" << std::endl;
}

void ProbabilityMap::add_Particles(){
    //adding particles

    constexpr float margin = 5.f;

    float max_x = std::clamp((origin_x_ + p_rad_), 0.f, static_cast<float>(width_));
    float min_x = std::clamp((origin_x_ - p_rad_), 0.f, static_cast<float>(width_));
    float max_y = std::clamp((origin_y_ + p_rad_), 0.f, static_cast<float>(height_));
    float min_y = std::clamp((origin_y_ - p_rad_), 0.f, static_cast<float>(height_));

    std::uniform_real_distribution<float> randomX(min_x + margin, max_x - margin);
    std::uniform_real_distribution<float> randomY(min_y + margin, max_y - margin);
    std::uniform_real_distribution<float> randomLife(50.f, 100.f);

    for(size_t i = 0; i < num_p_; i++){
        particles_.push_back( 
            PredictionParticle(randomX(rng_), randomY(rng_), randomLife(rng_)) 
        );
    };
}

void ProbabilityMap::computeProb(){
    //clear map from last frame
    std::fill(map_.begin(), map_.end(), 0.0f);
    //clear previous frames' occupied cell list
    occupied_cells.clear();

    //calculate width of cells
    std::size_t cols = width_ / cell_size_;


    for(size_t i = 0; i < particles_.size(); i++){
        //OOB check
        if(particles_[i].pos.x >= 0 && particles_[i].pos.x < width_ &&
        particles_[i].pos.y >= 0 && particles_[i].pos.y < height_){

            //calc pmap grid cell
            std::size_t cell_row = std::floor(particles_[i].pos.y / cell_size_);
            std::size_t cell_col = std::floor(particles_[i].pos.x / cell_size_);

            std::size_t cell_index = cell_row * cols + cell_col;

            //add new cell to occupied cells list
            if(map_[cell_index] == 0.f){
                occupied_cells.push_back(cell_index);
            }

            //add particle to cell map
            map_[cell_index] += 1.0f; 
                //change target cell if new update pushes cell to high probability than old target cell
                if(map_[cell_index] > map_[target_cell]){
                    target_cell = cell_index;
                }
                //std::cout << "target cell is: " << target_cell << std::endl;

        }
    }
};