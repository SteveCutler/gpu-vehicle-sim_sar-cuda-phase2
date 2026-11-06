#include "vehicle/dynamics.hpp"
#include <cmath>

Dynamics::Dynamics(){

};

VehicleState Dynamics::step_update(const VehicleState& vs, const Action& action, const environment& env, const float dt, std::size_t steps){

    //initial params
    constexpr float mass = 1.0f;
    constexpr float momentOfInertia = 1.0f;
    constexpr float linearDrag = 0.2f;
    constexpr float turnDrag = 0.35f;
    constexpr float pi = 3.14159265359f;

    //retrieve wind disturbance at this position
    const std::pair<float,float> current = env.getDisturbance(vs.x, vs.y, dt*steps);

    //Forces in x y coords
    const float thrustX = action.thrust * std::cos(vs.heading);
    const float thrustY = action.thrust * std::sin(vs.heading);

    //current is 0 for this first implementation
    const float relativeVx = vs.vx - current.first;
    const float relativeVy = vs.vy - current.second;

    //accel x and y
    const float ax = (thrustX - linearDrag * relativeVx) / mass;
    const float ay = (thrustY - linearDrag * relativeVy) / mass;

    //angular accel
    const float angularAcceleration = (action.torque - turnDrag * vs.turnRate) / momentOfInertia;

    //copy old state to preserve the things we aren't changing
    VehicleState newVs = vs;

    newVs.vx = vs.vx + ax * dt;
    newVs.vy = vs.vy + ay * dt;
    newVs.x = vs.x + newVs.vx * dt;
    newVs.y = vs.y + newVs.vy * dt;

    newVs.turnRate = vs.turnRate + angularAcceleration * dt;
    newVs.heading = std::remainder(vs.heading + newVs.turnRate * dt, 2.f * pi);

    return newVs;
}