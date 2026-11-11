#include "vehicle/SearchPlanner.hpp"
#include "vehicle/vehicleTypes.hpp"
#include <vector>
#include <set>
#include <algorithm>

SearchPlanner::SearchPlanner(std::size_t horizon, std::size_t v_num):
horizon_(horizon),
w_index_(0)
{
    // + 1 because the first index is the current position
    forecast_.resize(horizon_ + 1);
    #ifdef ENABLE_RENDERER
    preview.vehicles.resize(v_num);
    #endif
}

std::pair<float,float> SearchPlanner::chooseWaypoint(VehicleState vs, 
    const std::vector<PredictionParticle>& particles, 
    const environment& env, 
    const float dt,
    ProbabilityMap& tempMap,
    Dynamics& dynamics,
    Controller& controller,
    std::size_t height_,
    std::size_t vehicleIndex){
        //push back current waypoint as a candidate
        wayPoints.push_back({vs.goalx,vs.goaly});

        //init first step with current particle positions
        forecast_[0].resize(particles.size());
        forecast_[0] = particles;

        //build particle predictions vectors based on current fluid dynamics for all steps in horizon
        for(int step = 1; step <= horizon_; step++){
            //set size of current step
            forecast_[step].resize(particles.size());
            
            //loop through and advect particles into the future using current fluid dynamics step sample
            for(int p = 0; p < particles.size(); p++){
                std::pair<float,float> currentSample = env.getDisturbance(forecast_[step-1][p].pos.x, forecast_[step-1][p].pos.y, dt);
                forecast_[step][p] = forecast_[step-1][p];

                forecast_[step][p].pos.x += currentSample.first * dt;
                forecast_[step][p].pos.y += currentSample.second * dt;
            }

            
        }

        #ifdef ENABLE_RENDERER
        //All vehicles use the same particle forecast, so save this once per replan.
        if(vehicleIndex == 0){
            std::size_t sample_count = std::min<std::size_t>(32, particles.size());
            preview.particles.resize(sample_count);

            for(std::size_t p = 0; p < sample_count; p++){
                std::size_t particle_index = p * particles.size() / sample_count;
                auto& path = preview.particles[p];
                path.resize(horizon_ + 1);

                for(std::size_t step = 0; step <= horizon_; step++){
                    path[step] = {forecast_[step][particle_index].pos.x,
                                  forecast_[step][particle_index].pos.y};
                }
            }
        }
        #endif

        //build occupancy maps for various keyframes in horizon, 
        //calc waypoint halfway through horizon
        tempMap.particles_ = forecast_[horizon_/2];
        tempMap.computeProb();
        wayPoints.push_back(
            tempMap.convertCell(tempMap.target_cell)
        );

        //calc waypoint at horizon
        tempMap.particles_ = forecast_[horizon_];
        tempMap.computeProb();
        wayPoints.push_back(
            tempMap.convertCell(tempMap.target_cell)
        );

        #ifdef ENABLE_RENDERER
        auto& vehiclePreview = preview.vehicles[vehicleIndex];
        vehiclePreview.candidates.resize(wayPoints.size());
        for(auto& path : vehiclePreview.candidates){
            path.resize(horizon_ + 1);
        }
        #endif

        //reserve space for wp count
        wpCount_.resize(wayPoints.size());
        //init sim vehicle
        VehicleState simVehicle;
        Action action;
        std::size_t count;

        
        //calculate which waypoint yeilds highest scan of probability particles if the vehicle takes it
        for(int w = 0; w < wayPoints.size(); w++){
            //reset count
            std::set<std::size_t> seen;
            //reset boat state
            simVehicle = vs;
            //assign new wp;
            simVehicle.goalx = wayPoints[w].first;
            simVehicle.goaly = wayPoints[w].second;

            #ifdef ENABLE_RENDERER
            vehiclePreview.candidates[w][0] = {simVehicle.x, simVehicle.y};
            #endif

            for(int step = 1; step <= horizon_; step++){

                //calculate new state
                simVehicle = dynamics.step_update(simVehicle, controller.steer_controller(simVehicle), env, dt, step);

                #ifdef ENABLE_RENDERER
                vehiclePreview.candidates[w][step] = {simVehicle.x, simVehicle.y};
                #endif

                //scan for particles at this step within sensor range

                for(int p = 0; p < forecast_[step].size(); p++){
                    float x_pos = forecast_[step][p].pos.x;
                    float y_pos = forecast_[step][p].pos.y;

                    float x_dist = (x_pos - simVehicle.x);
                    float y_dist = (y_pos - (height_ - simVehicle.y));

                    if(x_dist*x_dist + y_dist*y_dist < (simVehicle.sensor_rad*2)*(2* simVehicle.sensor_rad)){
                        seen.insert(p);
                    }
                }


            }

            //decide which waypoint to pursue based on highest particle collisions
            wpCount_[w] = seen.size();

        }

        const auto best = std::max_element(wpCount_.begin(), wpCount_.end());
        const auto bestIndex = best - wpCount_.begin();
        const auto new_WP = wayPoints[bestIndex];

        #ifdef ENABLE_RENDERER
        vehiclePreview.selected = static_cast<std::size_t>(bestIndex);
        #endif

        //set new waypoint

        wayPoints.clear();
        return new_WP;

};
