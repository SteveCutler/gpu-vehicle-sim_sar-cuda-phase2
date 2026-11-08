#include "vehicle/environment_CUDA.hpp"
#include <cmath>

environment_CUDA::environment_CUDA(std::size_t w, std::size_t h, std::vector<Emitter*>& emitters, std::size_t seed): 
m_width(w), 
m_height(h),
velFieldx(w*h,0.0f), 
velFieldy(w*h,0.0f),
emitters(emitters),
seed(seed),
fluid_grid(FluidGrid_CUDA(m_width, m_height, emitters, seed, false)){

    //wind vel field  zero across the board for first implementation
    //in later implementation create curl noise field
    

    //initialize fluid grid
    fluid_grid = FluidGrid_CUDA(m_width, m_height, emitters, seed, false);


}

void environment_CUDA::update_fluid(float dt){
    fluid_grid.update(dt);
    return;
}

const std::vector<std::uint8_t> & environment_CUDA::get_pixels() const{
    
    return fluid_grid.m_pixels;
};

std::pair<float,float> environment_CUDA::getDisturbance(float x, float y, float time) const{
    {
        // bounds check
        x = std::max(0.f, std::min(x, m_width-1.f));
        y = std::max(0.f, std::min(y, m_height-1.f));

    return fluid_grid.vel_sample(x, y);
    }

}