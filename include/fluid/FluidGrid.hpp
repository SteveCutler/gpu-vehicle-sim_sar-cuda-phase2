#pragma once

#include <cstddef>
#include <vector>
#include "fluid/FastNoiseLite.h"
#include "fluid/Emitter.hpp"
#include <thread>
#include <algorithm>
#include <cstdint>
#include <utility>
#include "PerformanceClock.hpp"
#include "fluid/FluidTypes.hpp"


//Divergence measurement test struct
struct DivergenceStats {
    double rms =0.0;
    double maximumAbsolute = 0.0;
    bool allFinite = true;
};


class FluidGrid
    {
    // The CUDA backend uploads/downloads the existing CPU state.
    friend class CUDA_simulation;

        friend struct ScalarBoundaryTestAccess;

    public:
        FluidGrid(std::size_t width, std::size_t height, std::vector<Emitter*> emitters, int seed, bool threaded);
        std::vector<std::uint8_t> m_pixels;
        float m_buoyancy;
        float m_drift;
        float m_diff_co;
        float m_viscosity;
        float m_decay;
        float m_curl_mult;
        float m_noise_freq;
        bool m_mult_threaded;
        float m_noiseTimeMult;
        
        //multithreading by row template
        template <typename Func>
            void parallelForRows(
                std::size_t y_begin,
                std::size_t y_end,
                Func func){

                    //calc relevant variables for threading
                    std::size_t total_rows = y_end - y_begin;
                
                    //total rows to be processed, leaving out the boundaries
                    std::size_t rows_per_thread = (total_rows + m_thread_count-1)/m_thread_count;
                
                    //creating thread container and reserving space
                    std::vector<std::thread> threads;
                    threads.reserve(m_thread_count);
                
                    for ( std::size_t n = 0; n<m_thread_count; n++){
                        //calc start and end range
                        std::size_t start = y_begin+n*rows_per_thread;
                        std::size_t end_line = start+rows_per_thread;

                        //range check for end line
                        std::size_t end = std::min(end_line,y_end);
                
                        //range check
                        if(start>=end){
                            continue;
                        }
                
                        //start threads and add them to container
                        threads.emplace_back(
                                func,
                                start,
                                end
                            
                        );
                    }

                    //once all threads are created, wait until done
                    for(auto& thread : threads){
                        thread.join();
                    }
                }

        void update(float dt);
        
        void reset_density();

        //field getters
        const std::vector<float>& density_r() const;
        const std::vector<float>& density_g() const;
        const std::vector<float>& density_b() const;
        const std::vector<float>& u_velocity() const;
        const std::vector<float>& v_velocity() const;
        const std::vector<float>& get_noise() const;
        const std::vector<float>& pressure() const;
        const std::vector<float>& divergence() const;

        const std::vector<std::uint8_t> & get_pixels() const;

        // vel field setter for test purposes
        void setVelocityAt(std::size_t x, std::size_t y, float u,float v);

        //divergence stats test capture function
        [[nodiscard]] DivergenceStats measureDivergence() const;

        //moved to public for divergence test purposes
        void projectStep();


        //timing getters
        PerformanceClock m_performance_clock;
        float time_noise() const;
        float time_vel() const;
        float time_divergence() const;
        float time_pressure() const;
        float time_diffuse() const;
        float time_advect() const;
        float time_advectVel() const;
        float time_project() const;
        float time_addSource() const;

        std::pair<float,float> vel_sample(float x, float y) const;


    private:

        [[nodiscard]] float divergenceAt(std::size_t x, std::size_t y) const;

        size_t calcPos(size_t x, size_t y);
        
        void advect_decay_Threaded(float dt, const std::vector<float>& source, std::vector<float>& dest);
        void advect_decay_Rows(float dt, const std::vector<float>& source, std::vector<float>& dest, std::size_t begin, std::size_t end);

        void advectVel_Threaded(float dt);
        void advectVel_Rows(float dt, std::size_t begin, std::size_t end);

        
        
        void swapVel();
        
        void clearPressure();
        
        void calcNoise(float dt);
        void calcNoise_Threaded(float dt);
        void calcNoiseRows(float dt, std::size_t begin, std::size_t end);
        
        void calcVel_Rows(float dt, std::size_t begin, std::size_t end);
        void calcVel_Threaded(float dt);
        
        void calcDivergence_Threaded();
        void calcDivergence_Rows(std::size_t begin, std::size_t end);
        
        //single threaded pressure solve function
        void solvePressure();
        
        //multi threaded implementation
        void solvePressureRows(std::size_t begin, std::size_t end);
        void solvePressureThreaded();
        
        void velBoundaries();
        
        void pressureBoundaries();
        
        void project_Threaded();
        void project_Rows(std::size_t begin, std::size_t end);
        
        
        void addSource(size_t x, size_t y, float size, Vec3 clr);
        
        void swapDensity();
        
        void diffuse(float dt, const std::vector<float>& source, std::vector<float>& destination);
        void diffuseJacobi_Rows(float a, const std::vector<float>& source, const std::vector<float>& current, std::vector<float>& next, std::size_t begin, std::size_t end);
        void diffuse_Threaded(float a, const std::vector<float>& source, const std::vector<float>& current,  std::vector<float>& next);

        void diffuseVel_Threaded(float dt);
        void diffuseVel_Rows(float dt, std::size_t begin, std::size_t end);

        void densityToPixels();

        void gen_pixels_Threaded();
        void gen_pixels_Rows(std::size_t begin, std::size_t end);

        float density_sample(std::size_t i) const;
        
        void scalarBoundaries(std::vector<float>& field);
        
        FastNoiseLite m_noise_gen;




    
    private:

        //Global variables
        int m_seed;
        std::size_t m_width;
        std::size_t m_height;
        
        float m_vel_decay;
        float m_source;
        float m_elapsed;
        std::size_t m_pressure_iter;
        


        //Threading Variables
        std::size_t m_thread_count;
        

        //Timing Debug variables
        float m_noise_ms;
        float m_vel_ms;
        float m_div_ms;
        float m_pressure_ms;
        float m_advect_ms;
        float m_diffuse_ms;
        float m_advectVel_ms;
        float m_project_ms;
        float m_addSource_ms;
        float m_render_ms;

        //Emitters
        std::vector<Emitter*> m_emitters;

        // SoA member variables
        std::vector<float> m_density_r;
        std::vector<float> m_density_g;
        std::vector<float> m_density_b;
        std::vector<float> m_u_velocity;
        std::vector<float> m_v_velocity;

        std::vector<float> m_diffusion_scratch;
        std::size_t m_diffusion_iterations;

        std::vector<float>m_density_r_prev;
        std::vector<float>m_density_g_prev;
        std::vector<float>m_density_b_prev;
        std::vector<float>m_u_velocity_prev;
        std::vector<float>m_v_velocity_prev;

        std::vector<float> m_noise;

        std::vector<float> m_pressure;
        std::vector<float> m_pressure_prev;

        std::vector<float> m_divergence;

    };
