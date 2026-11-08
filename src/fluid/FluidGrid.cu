#include "fluid/FluidGrid_CUDA.hpp"
#include <vector>
#include<iostream>
#include <random>
#include <thread>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cassert>
#include <stdexcept>


FluidGrid_CUDA::FluidGrid_CUDA(std::size_t width, std::size_t height, std::vector<Emitter*> emitters, int seed, bool threaded):

m_pixels(width * height * 4, std::uint8_t{0}),
m_buoyancy(50.f),
m_drift(30.f),
m_diff_co(4.5f),
m_viscosity(0.05f),
m_decay(0.994f),
m_curl_mult(1000),
m_noise_freq(0.04f),
m_mult_threaded(threaded),
m_noiseTimeMult(10.f),

m_seed(seed),
m_width(width),
m_height(height),

m_vel_decay(0.9f),
m_source(50.f),
m_elapsed(0.f),
m_pressure_iter(20),

m_thread_count(4),

m_noise_ms(0.f),
m_vel_ms(0.f),
m_div_ms(0.f),
m_pressure_ms(0.f),
m_advect_ms(0.f),
m_diffuse_ms(0.f),
m_advectVel_ms(0.f),
m_project_ms(0.f),
m_addSource_ms(0.f),
m_render_ms(0.f),
m_emitters(emitters),

m_density_r(width*height, 0.0f),
m_density_g(width*height, 0.0f),
m_density_b(width*height, 0.0f),
m_u_velocity(width*height, 0.0f),
m_v_velocity(width*height, 0.0f),

m_diffusion_scratch(width*height,0.f),
m_diffusion_iterations{20},

m_density_r_prev(width*height, 0.0f),
m_density_g_prev(width*height, 0.0f),
m_density_b_prev(width*height, 0.0f),
m_u_velocity_prev(width*height, 0.0f),
m_v_velocity_prev(width*height, 0.0f),

m_noise(width*height, 0.0f),
m_pressure(width*height, 0.0f),
m_pressure_prev(width*height, 0.0f),
m_divergence(width*height, 0.0f)
{

    //configuration for random seed if desired
    // std::random_device rd;
    // std::mt19937 gen(rd()); 
    // std::uniform_int_distribution<int> distr(1, 100);

    //configure reproducible noise generator
    m_noise_gen.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    m_noise_gen.SetSeed(seed);

}

//field getters
const std::vector<float>& FluidGrid_CUDA::density_r() const{
    return m_density_r;
}
const std::vector<float>& FluidGrid_CUDA::density_g() const{
    return m_density_g;
}
const std::vector<float>& FluidGrid_CUDA::density_b() const{
    return m_density_b;
}

const std::vector<float>& FluidGrid_CUDA::u_velocity() const{
    return m_u_velocity;
};

const std::vector<float>& FluidGrid_CUDA::v_velocity() const{
    return m_v_velocity;
};

const std::vector<float>& FluidGrid_CUDA::get_noise() const{
    return m_noise;
};

const std::vector<std::uint8_t> & FluidGrid_CUDA::get_pixels() const{
    return m_pixels;
};

// timing getters

float FluidGrid_CUDA::time_noise() const {
    return m_noise_ms;
};

float FluidGrid_CUDA::time_vel() const{
    return m_vel_ms;
};

float FluidGrid_CUDA::time_divergence() const{
    return m_div_ms;
};

float FluidGrid_CUDA::time_pressure() const{
    return m_pressure_ms;
};

float FluidGrid_CUDA::time_diffuse() const{
    return m_diffuse_ms;
};

float FluidGrid_CUDA::time_advect() const{
    return m_advect_ms;
};

float FluidGrid_CUDA::time_advectVel() const{
    return m_advectVel_ms;
};

float FluidGrid_CUDA::time_project() const{
    return m_project_ms;
};

float FluidGrid_CUDA::time_addSource() const{
    return m_addSource_ms;
};


//UPDATE LOOP

void FluidGrid_CUDA::update(float dt){

    m_elapsed += dt;

    
    //Create noise scalar field
    m_performance_clock.restartMs();
    calcNoise(dt);

    m_noise_ms = m_performance_clock.restartMs();
    
    
    //Create vel gradient field based on noise
    m_performance_clock.restartMs();
    m_mult_threaded ? calcVel_Threaded(dt) : calcVel_Rows(dt, 1, m_height-1);

    m_vel_ms = m_performance_clock.restartMs();
    
    // apply velocity boundary calculations
    
    //Advect velocity
    swapVel();
    m_performance_clock.restartMs();
    m_mult_threaded ? advectVel_Threaded(dt) : advectVel_Rows(dt, 1,m_height-1);
    m_advectVel_ms = m_performance_clock.restartMs();
    
    velBoundaries();
    
    
    //Diffuse velocity
    swapVel();
    m_mult_threaded ? diffuseVel_Threaded(dt) : diffuseVel_Rows(dt,1,m_height-1);  
    velBoundaries();

    //second pressure solve
    PerformanceClock projectionClock;
    projectStep();
    m_project_ms = projectionClock.elapsedMs();
   // decayVel();  

    //Add density source
    m_performance_clock.restartMs();

    for( auto& emitter : m_emitters){

        addSource(emitter->m_pos.x, emitter->m_pos.y, emitter->m_rad, emitter->m_clr);
    }

    m_addSource_ms = m_performance_clock.restartMs();
    
    //Diffuse
    swapDensity();
    m_performance_clock.restartMs();
    diffuse(dt, m_density_r_prev, m_density_r);
    diffuse(dt, m_density_g_prev, m_density_g);
    diffuse(dt, m_density_b_prev, m_density_b);
    m_diffuse_ms = m_performance_clock.restartMs();
    
    //Advect
    swapDensity();
    m_performance_clock.restartMs();
    m_mult_threaded ? advect_decay_Threaded(dt, m_density_r_prev, m_density_r) : advect_decay_Rows(dt, m_density_r_prev, m_density_r, 1,m_height-1);
    m_mult_threaded ? advect_decay_Threaded(dt, m_density_g_prev, m_density_g) : advect_decay_Rows(dt, m_density_g_prev, m_density_g, 1,m_height-1);
    m_mult_threaded ? advect_decay_Threaded(dt, m_density_b_prev, m_density_b) : advect_decay_Rows(dt, m_density_b_prev, m_density_b, 1,m_height-1);
    scalarBoundaries(m_density_r);
    scalarBoundaries(m_density_g);
    scalarBoundaries(m_density_b);
    m_advect_ms = m_performance_clock.restartMs();

    //generate pixels
    #ifdef ENABLE_RENDERER
    m_mult_threaded ? gen_pixels_Threaded() : gen_pixels_Rows(0, m_height);
    #endif
    
}
    


//HELPERS

//measure divergence at x y
float FluidGrid_CUDA::divergenceAt(std::size_t x, std::size_t y) const{
    
    std::size_t row = m_width*y;
    std::size_t index = row+x;

    float xl = m_u_velocity[index-1];
    float xr = m_u_velocity[index+1];
    float yt = m_v_velocity[index-m_width];
    float yb = m_v_velocity[index+m_width];

    float div =  ((xr-xl)*.5)+((yb-yt)*.5);

    return div;
}

//measure divergence loop for test purposes
DivergenceStats_CUDA FluidGrid_CUDA::measureDivergence() const{

    DivergenceStats_CUDA stats;

    double squaredSum = 0.0;
    std::size_t sampleCount = 0;

    for(std::size_t x = 1; x<m_width-1; x++){
        for(std::size_t y = 1; y<m_height-1; y++){
            //calculating divergence at FluidGrid_CUDA cell
            const double value = static_cast<double>(divergenceAt(x,y));

            //finite number check
            if(!std::isfinite(value)){
                stats.allFinite = false;
                continue;
            }

            // adding to squared sum
            squaredSum +=value*value;

            //check if current value is greatest abs squared divergence
            stats.maximumAbsolute = std::max(stats.maximumAbsolute, std::abs(value));

            sampleCount ++;

        }

        if(sampleCount>0){
            stats.rms=std::sqrt(squaredSum/static_cast<double>(sampleCount));
        }

        
    }
    return stats;

}

void FluidGrid_CUDA::setVelocityAt(std::size_t x, std::size_t y, float u, float v){

    if(x>=m_width || y>=m_height){
        throw std::out_of_range("Velocity position outside of FluidGrid_CUDA");
    }

    const std::size_t index = y*m_width +x;
    
    m_u_velocity[index] = u;
    m_v_velocity[index] = v;
}

float FluidGrid_CUDA::density_sample(std::size_t i) const{
    const float density = std::max({
        m_density_r[i],
        m_density_g[i],
        m_density_b[i]
    });

    return std::clamp(density, 0.f, 1.f);

}

//helper for handling boundaries of any scalar field
void FluidGrid_CUDA::scalarBoundaries(std::vector<float>& field){

    assert(m_width >= 2);
    assert(m_height >= 2);
    assert(field.size() == m_width * m_height);

    //left and right boundaries
    for(std::size_t y = 1; y<m_height -1; y++){
        const std::size_t row = y*m_width;
        field[row] = field[row + 1];
        field[row + m_width-1] = field[row+m_width-2];
    }

    //top and bottom boundaries
    for(std::size_t x =1; x<m_width-1; x++){
        field[x] = field[x+m_width];
        field[(m_height-1)*m_width +x] = field[(m_height-2)*m_width+x];
    }

    //top left corner
    field[0] = (field[1]+field[m_width])*.5f;

    //top right corner
    field[m_width-1] = (field[(m_width*2-1)]+field[m_width-2])*.5f;

    //bottom left corner
    field[m_width*(m_height-1)] = (field[m_width*(m_height-1)+1]+field[m_width*(m_height-2)])*.5f;

    //bottom right corner
    field[m_width*m_height-1] = (field[m_width*m_height-2]+field[m_width*m_height-m_width-1])*.5f;
}

void FluidGrid_CUDA::reset_density(){
    std::fill(m_density_r.begin(), m_density_r.end(), 0.0f);
    std::fill(m_density_g.begin(), m_density_g.end(), 0.0f);
    std::fill(m_density_b.begin(), m_density_b.end(), 0.0f);
    std::fill(m_density_r_prev.begin(), m_density_r_prev.end(), 0.0f);
    std::fill(m_density_g_prev.begin(), m_density_g_prev.end(), 0.0f);
    std::fill(m_density_b_prev.begin(), m_density_b_prev.end(), 0.0f);

    std::fill(m_u_velocity.begin(), m_u_velocity.end(), 0.0f);
    std::fill(m_v_velocity.begin(), m_v_velocity.end(), 0.0f);
    std::fill(m_u_velocity_prev.begin(), m_u_velocity_prev.end(), 0.0f);
    std::fill(m_v_velocity_prev.begin(), m_v_velocity_prev.end(), 0.0f);

    std::fill(m_pressure.begin(), m_pressure.end(), 0.0f);
    std::fill(m_pressure_prev.begin(), m_pressure_prev.end(), 0.0f);
    std::fill(m_divergence.begin(), m_divergence.end(), 0.0f);

};
void FluidGrid_CUDA::projectStep(){
    //clearPressure();
    m_performance_clock.restartMs();
    if(m_mult_threaded){
        calcDivergence_Threaded();
    }
    else{
        calcDivergence_Rows(1,m_height-1);
    }

    m_div_ms = m_performance_clock.restartMs();


    m_performance_clock.restartMs();

    //switch for turning on multithreaded mode
    m_mult_threaded ? solvePressureThreaded() : solvePressure();

    m_pressure_ms = m_performance_clock.restartMs();
    pressureBoundaries();
    
    //m_performance_clock.restartMs();
    if(m_mult_threaded){
        project_Threaded();
    }
    else{
        project_Rows(1,m_height-1);
    }

    velBoundaries();

}

void FluidGrid_CUDA::swapVel(){
    std::swap(m_u_velocity, m_u_velocity_prev);
    std::swap(m_v_velocity, m_v_velocity_prev);
};

void FluidGrid_CUDA::velBoundaries(){
    //set vel on left and right boundaries
    for(std::size_t y = 0; y<m_height; y++){
        std::size_t b_l = y*m_width;
        std::size_t b_r = (m_width-1)+y*m_width;

        //horizontal vel 0
        m_u_velocity[b_l] = 0.f;
        m_u_velocity[b_r] = 0.f;
        //vertical vel refers to neighbour
        m_v_velocity[b_l] = m_v_velocity[b_l+1];
        m_v_velocity[b_r] = m_v_velocity[b_r-1];
    }

    //set vel for top and bottom boundaries
    for(std::size_t x = 0; x<m_width; x++){
        std::size_t b_t = x;
        std::size_t b_b = x+(m_height-1)*m_width;

        //vertical vel refers to neighbour
        m_u_velocity[b_t] = m_u_velocity[b_t+m_width];
        m_u_velocity[b_b] = m_u_velocity[b_b-m_width];
        //vertical vel 0
        m_v_velocity[b_t] =0.f;
        m_v_velocity[b_b] = 0.f;
    }
}

void FluidGrid_CUDA::pressureBoundaries(){
    //set pressure on left and right boundaries
    for(std::size_t y = 0; y<m_height; y++){
        std::size_t b_l = y*m_width;
        std::size_t b_r = (m_width-1)+y*m_width;


        //pressure refers to x
        m_pressure[b_l] = m_pressure[b_l+1];
        m_pressure[b_r] = m_pressure[b_r-1];
    }

    //set pressure for top and bottom boundaries
    for(std::size_t x = 0; x<m_width; x++){
        std::size_t b_t = x;
        std::size_t b_b = x+(m_height-1)*m_width;

        //refers to vertical neighbour
        m_pressure[b_t] = m_pressure[b_t+m_width];
        m_pressure[b_b] = m_pressure[b_b-m_width];

    }
}

void FluidGrid_CUDA::clearPressure(){
    std::fill(m_pressure.begin(),m_pressure.end(),0.f);
    std::fill(m_pressure_prev.begin(),m_pressure_prev.end(),0.f);
};


void FluidGrid_CUDA::calcNoise(float dt){
    m_elapsed += dt*m_noiseTimeMult;   

    m_noise_gen.SetFrequency(m_noise_freq);

    const float sample_time = m_elapsed;

    if(m_mult_threaded){
        calcNoise_Threaded(sample_time);
    }
    else{
        calcNoiseRows(sample_time,0,m_height);
    }
}

void FluidGrid_CUDA::calcNoise_Threaded(float sample_time){
    

    parallelForRows(0, m_height,
        [this, sample_time](std::size_t begin, std::size_t end){
            calcNoiseRows(sample_time, begin, end);
        });

};

void FluidGrid_CUDA::calcNoiseRows(float dt, std::size_t begin, std::size_t end){

    std::size_t index = begin*m_width;

    for(std::size_t y = begin; y<end; y++){
         for(std::size_t x = 0; x<m_width; x++){
            
            float n = m_noise_gen.GetNoise(x*1.f,y*1.f, dt);
            m_noise[index] = n;
            index++;
        }
    }
};

void FluidGrid_CUDA::calcVel_Threaded(float dt){
    FluidGrid_CUDA::parallelForRows(
        1,
        m_height-1,
        [this, dt](std::size_t begin, std::size_t end){
            calcVel_Rows(dt, begin, end);
        }
    );
}

void FluidGrid_CUDA::calcVel_Rows(float dt, std::size_t begin, std::size_t end){
    for(std::size_t y = begin; y<end; y++){
        std::size_t row = y*m_width;

         for(std::size_t x = 1; x<m_width-1; x++){
            std::size_t index = row+x;



                float xl = m_noise[index-1];
                float xr = m_noise[index+1];
                float yt = m_noise[index-m_width];
                float yb = m_noise[index+m_width];

                float dx = (xr-xl)/2;
                float dy = (yb-yt)/2;

                m_u_velocity[index] += (-dy*m_curl_mult*dt);
                m_v_velocity[index] += (dx*m_curl_mult*dt);
               
            

        }
    }
}


void FluidGrid_CUDA::swapDensity(){
    std::swap(m_density_r, m_density_r_prev);
    std::swap(m_density_g, m_density_g_prev);
    std::swap(m_density_b, m_density_b_prev);
}


void FluidGrid_CUDA::addSource(size_t x, size_t y, float size, Vec3 clr){

   
    for(int dy = -size; dy<=size; dy++){
        //new y
        std::size_t new_y = (y+dy);
        std::size_t row = new_y*m_width;
        
        for (int dx = -size; dx <=size; dx++){
            //new_x
            std::size_t new_x = x+dx;


            if(new_x>1 && new_x<m_width-1
            && new_y>1 && new_y<m_height-1){

                
                float dist = (dx*dx) + (dy*dy);
                float rad_sqr = size*size;
                
                if(dist < rad_sqr){
                    float density_r = clr.x * (rad_sqr-dist)/rad_sqr;
                    float density_g = clr.y * (rad_sqr-dist)/rad_sqr;
                    float density_b = clr.z * (rad_sqr-dist)/rad_sqr;

                    size_t pos = new_x+row;
                    m_density_r[pos] = std::clamp((m_density_r[pos]+density_r),0.f,1.f);
                    m_density_g[pos] = std::clamp((m_density_g[pos]+density_g),0.f,1.f);
                    m_density_b[pos] = std::clamp((m_density_b[pos]+density_b),0.f,1.f);
                }
            }

        }
    }
}


//DYNAMIC VEL FUNCTIONS

void FluidGrid_CUDA::advectVel_Threaded(float dt){
    FluidGrid_CUDA::parallelForRows(
        1,
        m_height-1,
        [this, dt](std::size_t begin, std::size_t end){
            advectVel_Rows(dt, begin, end);
        }
    );
    

}; 

void FluidGrid_CUDA::advectVel_Rows(float dt, std::size_t begin, std::size_t end){
    float new_u_vel = 0.f;
    float new_v_vel = 0.f;

    //switch to row first y x loop
    for(std::size_t y = begin; y<end; y++){
        
        //reducing multiplication calculations by doing it once per row
        std::size_t row = y*m_width;

        for(std::size_t x = 0; x<m_width; x++){

        //addition calculations are cheap for CPU
        std::size_t i = row + x;
    
        //index velocity
        float old_u_vel = m_u_velocity_prev[i];
        float old_v_vel = m_v_velocity_prev[i];

        //removed modulo and multiplicatin operations happening for every cell

        //backwards location lookup 
        float new_u_pos = x - old_u_vel*dt;
        float new_v_pos = y - old_v_vel*dt;

        //finding 4 corners for bilinear interpolation, clamping in boundaries
        float x_sample = std::clamp(new_u_pos,1.0f,m_width-2.0f);
        float y_sample = std::clamp(new_v_pos,1.0f,m_height-2.0f);

        std::size_t x_floor = std::floor(x_sample);
        std::size_t x_high = x_floor+1;
        std::size_t y_floor = std::floor(y_sample);
        std::size_t y_high = y_floor+1;

        //finding values at corners
        std::size_t tl_index = x_floor+y_floor*m_width;
        std::size_t tr_index = x_high+y_floor*m_width;
        std::size_t bl_index = x_floor+y_high*m_width;
        std::size_t br_index = x_high+y_high*m_width;

        float tl_u = m_u_velocity_prev[tl_index];
        float tr_u = m_u_velocity_prev[tr_index];
        float bl_u = m_u_velocity_prev[bl_index];
        float br_u = m_u_velocity_prev[br_index];
        
        float tl_v = m_v_velocity_prev[tl_index];
        float tr_v = m_v_velocity_prev[tr_index];
        float bl_v = m_v_velocity_prev[bl_index];
        float br_v = m_v_velocity_prev[br_index];

        //distance from edges
        float dx = x_sample - x_floor;
        float dy = y_sample - y_floor;

        //weighting calculations
        float tl_weight = (1-dx)*(1-dy);
        float tr_weight = dx*(1-dy);
        float bl_weight = (1-dx)*dy;
        float br_weight = dx*dy;

        new_u_vel = (tl_u*tl_weight + tr_u*tr_weight + bl_u*bl_weight + br_u*br_weight) *m_vel_decay;
        new_v_vel = (tl_v*tl_weight + tr_v*tr_weight + bl_v*bl_weight + br_v*br_weight) *m_vel_decay;
        new_u_vel -= m_drift * density_sample(i) * dt ;
        new_v_vel -= m_buoyancy * density_sample(i) * dt ;
        
        m_u_velocity[i] = new_u_vel;
        m_v_velocity[i] = new_v_vel;

        }
    }
}

void FluidGrid_CUDA::diffuseVel_Threaded(float dt){
    FluidGrid_CUDA::parallelForRows(
        1,
        m_height-1,
        [this, dt](std::size_t begin, std::size_t end){
            diffuseVel_Rows(dt, begin, end);
        }    
    );    
}    

void FluidGrid_CUDA::diffuseVel_Rows(float dt, std::size_t begin, std::size_t end){

    float left_u = 0.f;
    float right_u = 0.f;
    float up_u = 0.f;
    float down_u = 0.f;
    float center_u = 0.f;

    float left_v = 0.f;
    float right_v = 0.f;
    float up_v = 0.f;
    float down_v = 0.f;
    float center_v = 0.f;

    float lap_u = 0.f;
    float lap_v = 0.f;

    float new_u_vel = 0.f;
    float new_v_vel = 0.f;

    for(std::size_t y = begin; y<end; y++){
        //minimize mult operations
        std::size_t row = y*m_width;

        for(std::size_t x = 1; x<m_width-1; x++){
            //calculate indexx
            std::size_t i = row+x; 



                //calc corner indices

                std::size_t l_index = i-1;
                std::size_t r_index = i+1;
                std::size_t u_index = i-m_width;
                std::size_t d_index = i+m_width;

                //getting data for laplacian
                left_u = m_u_velocity_prev[l_index];
                right_u = m_u_velocity_prev[r_index];
                up_u = m_u_velocity_prev[u_index];
                down_u = m_u_velocity_prev[d_index];
                center_u = m_u_velocity_prev[i];

                left_v = m_v_velocity_prev[l_index];
                right_v = m_v_velocity_prev[r_index];
                up_v = m_v_velocity_prev[u_index];
                down_v = m_v_velocity_prev[d_index];
                center_v = m_v_velocity_prev[i];

                //calculate laplacian
                lap_u = left_u+right_u+up_u+down_u - (4*center_u);
                lap_v = left_v+right_v+up_v+down_v - (4*center_v);

                //calculate and write new density
                new_u_vel = m_u_velocity_prev[i]+ m_viscosity * lap_u*dt;
                new_v_vel = m_v_velocity_prev[i]+ m_viscosity * lap_v*dt;

                m_u_velocity[i] = new_u_vel;
                m_v_velocity[i] = new_v_vel;

                

        }
    }
};


void FluidGrid_CUDA::gen_pixels_Threaded(){
    FluidGrid_CUDA::parallelForRows(
        0,
        m_height,
        [this](std::size_t begin, std::size_t end){
            gen_pixels_Rows(begin, end);
        }    
    );    
}    

void FluidGrid_CUDA::gen_pixels_Rows(std::size_t begin, std::size_t end){
            //Convert density buffer to pixel data
        for (std::size_t y = begin; y<end; y++){
            std::size_t row = y*m_width;

            for( std::size_t x=0; x<m_width; x++){
                std::size_t i = row+x;

                auto [u,v] = vel_sample(x,y);
                float speed = std::hypot(u,v);
                float white_level = 50.0;
                float current = std::clamp(speed/white_level, 0.f, 1.f);

                
                //clamp between 0 and 1
                float d_r = 0.1f + current * 0.9f;
                float d_g = 0.45f + current * 0.55f;
                float d_b = 0.9f + current * 0.1f;
                
                
                //convert density to RGB values
                std::uint8_t value_r = static_cast<std::uint8_t>(d_r * 255.f);
                std::uint8_t value_g = static_cast<std::uint8_t>(d_g * 255.f);
                std::uint8_t value_b = static_cast<std::uint8_t>(d_b * 255.f);
                
                //find correct position in pixel array, given each pixel has 1 components
                std::size_t p = i * 4;
                
                //create greyscale image with alpha of 1
                m_pixels[p] = value_r;  //R
                m_pixels[p+1] = value_g;//G
                m_pixels[p+2] = value_b;//B
                m_pixels[p+3] = 255;  //Alpha channel
            }            
    }        
};    






void FluidGrid_CUDA::calcDivergence_Threaded(){
    FluidGrid_CUDA::parallelForRows(
        1,
        m_height-1,
        [this](std::size_t begin, std::size_t end){
            calcDivergence_Rows(begin, end);
        }
    );
}

void FluidGrid_CUDA::calcDivergence_Rows(std::size_t begin, std::size_t end){

   
    
    for(std::size_t y =begin; y<end; y++){
        //calc row once per line
        std::size_t row = y*m_width;

            for(std::size_t x = 1; x<m_width-1; x++){


                    std::size_t index = row+x;

                    float xl = m_u_velocity[index-1];
                    float xr = m_u_velocity[index+1];
                    float yt = m_v_velocity[index-m_width];
                    float yb = m_v_velocity[index+m_width];

                    float div =  ((xr-xl)*.5)+((yb-yt)*.5);

                   

                    m_divergence[index] = div;
                
                
            }
        }

};

void FluidGrid_CUDA::project_Threaded(){
    FluidGrid_CUDA::parallelForRows(
        1,
        m_height-1,
        [this](std::size_t begin, std::size_t end){
            project_Rows(begin, end);
        }
    );
}

void FluidGrid_CUDA::project_Rows(std::size_t begin, std::size_t end){
    
    for(std::size_t y = begin; y<end; y++){
        std::size_t row = y*m_width;
        
        for(std::size_t x = 1; x<m_width-1; x++){
            
            
            std::size_t index = x+row;
            
            float xl = m_pressure[index-1];
            float xr = m_pressure[index+1];
            float yt = m_pressure[index-m_width];
            float yb = m_pressure[index+m_width];
            
            float pres_x =  (xr-xl)/2.f;
            float pres_y =  (yb-yt)/2.f;
            
            m_u_velocity[index] -= pres_x;
            m_v_velocity[index] -= pres_y;
            
            
        }
    }
    
};


void FluidGrid_CUDA::solvePressure(){
    //one iteratio of pressure solve
    std::size_t k = 0;
    while (k<m_pressure_iter){

        //exclude top boundaries
        for(std::size_t y = 1; y<m_height-1; y++){
    
            //calc row once per line
            std::size_t row = y*m_width;
    

                for(std::size_t x = 1; x<m_width-1; x++){
    
                        std::size_t index = x+row;
    
                        float xl = m_pressure_prev[index-1];
                        float xr = m_pressure_prev[index+1];
                        float yt = m_pressure_prev[index-m_width];
                        float yb = m_pressure_prev[index+m_width];
    
                        float pressure =  (xl+xr+yt+yb - m_divergence[index])*0.25f;
    
                        m_pressure[index] = pressure;
                    
                    
                }
            }
            k++;
            pressureBoundaries();
            std::swap(m_pressure, m_pressure_prev);
        }
        std::swap(m_pressure, m_pressure_prev);


};

void FluidGrid_CUDA::solvePressureRows(std::size_t begin, std::size_t end){

    //one iteratio of pressure solve

    for(std::size_t y = begin; y<end; y++){

        //calc row once per line
        std::size_t row = y*m_width;

            for(std::size_t x = 1; x<m_width-1; x++){

                    std::size_t index = x+row;

                    float xl = m_pressure_prev[index-1];
                    float xr = m_pressure_prev[index+1];
                    float yt = m_pressure_prev[index-m_width];
                    float yb = m_pressure_prev[index+m_width];

                    float pressure =  (xl+xr+yt+yb - m_divergence[index])*0.25f;

                    m_pressure[index] = pressure;
                
                
            }
        }

};

void FluidGrid_CUDA::solvePressureThreaded(){
    std::size_t k = 0;

    while(k<m_pressure_iter){

        parallelForRows( 
            1, m_height-1,
            [this](std::size_t start, std::size_t end){
                solvePressureRows(start, end);

            }
        );

        //then proceed with end of iteration loop like normal
        pressureBoundaries();
        k++;
        std::swap(m_pressure,m_pressure_prev);
    }

    //swap back to most recent pressure FluidGrid_CUDA on last iteration, optimize this away at some point
    std::swap(m_pressure,m_pressure_prev);

};

void FluidGrid_CUDA::diffuse(float dt, const std::vector<float>& source, std::vector<float>& destination){
const float a = m_diff_co * dt;

//take source field as initial guess
destination = source;

std::vector<float>* current = &destination;
std::vector<float>* next = &m_diffusion_scratch;

//iteration loop
for(std::size_t iteration = 0; iteration < m_diffusion_iterations; iteration++){

    if(m_mult_threaded){
        FluidGrid_CUDA::parallelForRows(
        1,
        m_height-1,
        [this, a, &source, current, next](std::size_t begin, std::size_t end){
                diffuseJacobi_Rows(a, source, *current, *next, begin, end);
                }
            );
        }
        else{
            diffuseJacobi_Rows(a, source, *current, *next, 1, m_height -1);

        }

        scalarBoundaries(*next);
        std::swap(current, next);
    }
    if (current != &destination) {
        destination.swap(m_diffusion_scratch);
    }
}



void FluidGrid_CUDA::diffuseJacobi_Rows(float a, const std::vector<float>& source, const std::vector<float>& current,  std::vector<float>& next, std::size_t begin, std::size_t end){

    //5 points diffusion kernel

    const float denominator = 1.f + 4.f * a;

    for(std::size_t y = begin; y<end; y++){

        const std::size_t row = y*m_width;

        for(std::size_t x = 1; x<m_width-1; x++){

            const std::size_t i = row+x;

            const float neighbourSum = 
                current[i-1] +
                current[i+1] + 
                current[i-m_width] + 
                current[i+m_width];

            next[i] = 
            (source[i] + a * neighbourSum)/ denominator;                

            }
        }
}


void FluidGrid_CUDA::advect_decay_Threaded(float dt, const std::vector<float>& source, std::vector<float>& dest){
    FluidGrid_CUDA::parallelForRows(
        1,
        m_height-1,
        [this, dt, &source, &dest](std::size_t begin, std::size_t end){
            advect_decay_Rows(dt, source, dest, begin, end);
        }
    );
    
    
}

void FluidGrid_CUDA::advect_decay_Rows(float dt, const std::vector<float>& source, std::vector<float>& dest, std::size_t begin, std::size_t end){


    float new_dens = 0.f;

    for(std::size_t y = begin; y<end; y++){
        std::size_t row = y*m_width;
        for(std::size_t x = 0; x<m_width; x++){
            std::size_t i =row+x;
            //index velocity
            float v_vel = m_v_velocity[i];
            float u_vel = m_u_velocity[i];


            //backwards location lookup 
            float new_x = std::clamp((x - u_vel*dt),1.f,m_width-2.f);
            float new_y = std::clamp((y - v_vel*dt),1.f,m_height-2.f);

            //finding 4 corners for bilinear interpolation, clamping in boundaries
            std::size_t x_low = std::floor(new_x);
            std::size_t x_high = x_low+1;
            std::size_t y_low =std::floor(new_y);
            std::size_t y_high = y_low+1;

            //finding values at corners
            float tl = source[x_low+y_low*m_width];
            float tr = source[x_high+y_low*m_width];
            float bl = source[x_low+y_high*m_width];
            float br = source[x_high+y_high*m_width];

            //distance from edges
            float dx = new_x - x_low;
            float dy = new_y - y_low;

            //weighting calculations
            float tl_weight = (1-dx)*(1-dy);
            float tr_weight = dx*(1-dy);
            float bl_weight = (1-dx)*dy;
            float br_weight = dx*dy;

            new_dens = tl*tl_weight + tr*tr_weight + bl*bl_weight + br*br_weight;

            dest[i] = new_dens*m_decay;

        }

    }

}


std::pair<float,float> FluidGrid_CUDA::vel_sample(float x, float y) const{
        float u_vel = 0.f;
        float v_vel = 0.f;

        std::size_t x_floor = std::floor(x);
        std::size_t x_high = x_floor+1;
        std::size_t y_floor = std::floor(y);
        std::size_t y_high = y_floor+1;

        std::size_t i = y_floor*m_width + x_floor;

        //finding values at corners
        std::size_t tl_index = x_floor+y_floor*m_width;
        std::size_t tr_index = x_high+y_floor*m_width;
        std::size_t bl_index = x_floor+y_high*m_width;
        std::size_t br_index = x_high+y_high*m_width;

        float tl_u = m_u_velocity_prev[tl_index];
        float tr_u = m_u_velocity_prev[tr_index];
        float bl_u = m_u_velocity_prev[bl_index];
        float br_u = m_u_velocity_prev[br_index];
        
        float tl_v = m_v_velocity_prev[tl_index];
        float tr_v = m_v_velocity_prev[tr_index];
        float bl_v = m_v_velocity_prev[bl_index];
        float br_v = m_v_velocity_prev[br_index];

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
