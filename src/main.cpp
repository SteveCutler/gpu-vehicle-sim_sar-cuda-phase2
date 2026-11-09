//Vehicle Include
#include <cstddef>
#include "vehicle/environment.hpp"
#include "vehicle/vehicleBatch.hpp"
#include "vehicle/controller.hpp"
#include "vehicle/dynamics.hpp"
#include <chrono>
#include <iostream>
#include "vehicle/stateExport.hpp"
#include "vehicle/DriftItem.hpp"
#include "vehicle/DriftDynamics.hpp"
#include "vehicle/ProbabilityMap.hpp"
#include "vehicle/SearchPlanner.hpp"


//Fluid Include
#include <iostream>
#include <vector>

#include "fluid/FluidGrid.hpp"
#include "fluid/FluidTypes.hpp"
#include "fluid/Emitter.hpp"

#ifdef ENABLE_RENDERER
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include "vehicle/renderer.hpp"
#endif

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "vehicle/CUDA_simulation.hpp"
#endif

using Clock = std::chrono::steady_clock;

int main(int argNum, char* argVals[]){


    std::cout << "starting up..." << std::endl;
    //master variables
    constexpr std::size_t width = 500;
    constexpr std::size_t height = 500;
    constexpr float dt = 0.02f;
    constexpr bool arrow_viz = false;
    constexpr std::size_t seed = 42;


    //CUDA SETTINGS
    bool useCuda = false;

    #ifdef ENABLE_CUDA
        useCuda = true;

    #endif

     ////////////
    //FLUID INIT
    ////////////

        Emitter blue{Vec2{width*.5f,height*.5f}, width*1.0f, Vec3{0.f,0.f,.2f}};


        std::vector<Emitter*> emitters{
            &blue,

        };
        

    //mutlithreaded switch
    bool threaded = false;

    //Font loading
    // sf::Font font;
    // bool loaded = font.openFromFile(FLUID_FONT_PATH);

    // if(!loaded){
    //     std::cerr << "Failed to load font\n";
    //     return EXIT_FAILURE;
    // }

    //Vehicles
    std::size_t N = 2;
    //Steps
    std::size_t steps = -1;

    //take in input variables
    //set vehicle batch size
    if (argNum > 1) N = std::stoi(argVals[1]);
    //set step size
    if (argNum > 2) steps = std::stoi(argVals[2]);
    
   // std::cout << "Vehicles: " << N << "\nSteps: " << steps << '\n';
    
    std::size_t curr_step = 0;
    
    //initialize environment with 0 wind at first
    environment env(width, height, emitters, seed);

    //target initialization
    DriftItem target;
    target.x = static_cast<float>(width/2.0f);
    target.y = static_cast<float>(height*0.8f);

    //initialize vehicleState data
    vehicleBatch vehicles(N, env, target.x, height - target.y);
    
    //create controller object
    Controller controller;

    //create dynamics updater
    Dynamics dynamics;
    //Drift dynamics
    DriftDynamics driftDynamics;

    //init Probability Map
    float prob_rad = 25.f;
    std::size_t particle_count = 10000;
    std::size_t cell_size = 50;

    //convert target spawn location to cell to initialize vehicles to target last known location of object

    ProbabilityMap p_map(driftDynamics, width, height, particle_count, target.x, target.y, prob_rad, cell_size);
    //used for calculating future waypoints
    ProbabilityMap tempMap(driftDynamics, width, height, particle_count, target.x, target.y, prob_rad, cell_size);
        //compute probability to generate target cell
    p_map.computeProb();


    //allocating variables outside hot loop
    VehicleState vs;
    Action action;
    VehicleState newState;
    //init planner, horizon of 50 steps
    SearchPlanner planner(50);

   

    // create a clock to track the elapsed time
    // TO DO create performance clock

    bool target_found = false;

    //if rendering enabled create rendering logic
    #ifdef ENABLE_RENDERER
        //create renderer
        Renderer renderer(width, height, arrow_viz, cell_size);
    #endif

    #ifdef ENABLE_CUDA
        CUDA_simulation CudaSim(planner, target, p_map, tempMap, env, N, dt);
        CudaSim.init_simulation(
            vehicles
        );
    #endif


        //start wallclock timer for performance measurement
        const auto start = Clock::now();

        //run sim for steps amount of steps
        if(useCuda){
                //move work to GPU
                #ifdef ENABLE_CUDA
                   curr_step = CudaSim.runCudaSimulation(vehicles, steps);
                #endif
            }
        else{
            while(curr_step < steps){
                // Advance the entire batch before scanning, matching the GPU kernel.
                for(std::size_t x = 0; x < N; ++x){
                    vs = vehicles.load(x);
                    action = controller.steer_controller(vs);
                    newState = dynamics.step_update(vs, action, env, dt, curr_step);
                    vehicles.set(x, newState);
                }

                // Process sensor updates in the same vehicle order as the CUDA path.
                for(std::size_t x = 0; x < N; ++x){
                    newState = vehicles.load(x);
                    target_found = vehicles.scan(newState, p_map, target);

                    if(target_found){
                        vehicles.set(x, newState);
                        break;
                    }

                    if(curr_step % 50 == 0 && curr_step != 0){
                        const auto newWP = planner.chooseWaypoint(
                            newState, p_map.particles_, env, dt, tempMap,
                            dynamics, controller, height);
                        newState.goalx = newWP.first;
                        newState.goaly = newWP.second;
                    }
                    vehicles.set(x, newState);
                }

                // All vehicles advanced once, even when this step finds the target.
                if(target_found){
                    ++curr_step;
                    break;
                }

                driftDynamics.DriftUpdate(
                    target, env.getDisturbance(target.x, target.y, 0.0f), dt);
                p_map.update(env, dt);
                p_map.computeProb();
                env.update_fluid(dt);
                ++curr_step;

                #ifdef ENABLE_RENDERER
                    if(!renderer.draw(vehicles, dt, curr_step, env, target, p_map)){
                        break;
                    }
                #endif
            }
        }

        //end clock
        const auto stop = Clock::now();

        //calculate elapsed time
        const double seconds = 
            std::chrono::duration<double>(stop - start).count();

        //multiply steps by vehicles to get total updates
        const double updates = static_cast<double>(N) * curr_step;

        std::cout << "Vehicles: " << N << '\n';
        std::cout << "Completed steps: " << curr_step << '\n';
        std::cout << "Execution time: " << seconds << " seconds\n";
        std::cout << "Vehicle updates/sec: " << updates / seconds << '\n';

            //output file name for correctness comparison
            if (argNum > 3) {
                exportStates(vehicles, argVals[3]);

                exportParticles(
                    p_map.particles_,
                    std::string(argVals[3]) + ".particles.csv");
            }
            

    // Keep showing the final state and handling window events.
    #ifdef ENABLE_RENDERER
    while (renderer.draw(vehicles, dt, curr_step, env, target, p_map)) {}
    #endif

 

    return 0;
}
