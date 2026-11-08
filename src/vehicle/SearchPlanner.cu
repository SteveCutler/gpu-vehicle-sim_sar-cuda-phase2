#include "vehicle/SearchPlanner_CUDA.hpp"
#include "vehicle/vehicleTypes.hpp"
#include <vector>
#include <cuda_runtime.h>
#include <set>
#include <algorithm>

SearchPlanner_CUDA::SearchPlanner_CUDA(std::size_t horizon):
horizon_(horizon),
w_index_(0)
{
    // + 1 because the first index is the current position
    forecast_.resize(horizon_);
}

std::pair<float,float> SearchPlanner_CUDA::chooseWaypoint(VehicleState vs, 
    const std::vector<PredictionParticle>& particles, 
    const environment& env, 
    const float dt,
    ProbabilityMap& tempMap,
    DynamicsCuda& dynamics,
    ControllerCuda& controller,
    std::size_t height_){
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

            for(int step = 1; step <= horizon_; step++){

                //calculate new state
                simVehicle = dynamics.step_update(simVehicle, controller.steer_controller(simVehicle), env, dt, step);

                //scan for particles at this step within sensor range

                for(int p = 0; p < forecast_[step].size(); p++){
                    float x_pos = forecast_[step][p].pos.x;
                    float y_pos = forecast_[step][p].pos.y;

                    float x_dist = abs(x_pos - simVehicle.x);
                    float y_dist = abs(y_pos - (height_ - simVehicle.y));

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

        //set new waypoint

        wayPoints.clear();
        return new_WP;

};
