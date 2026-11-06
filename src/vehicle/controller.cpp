#include "vehicle/controller.hpp"
#include <cmath>
#include <algorithm>

Controller::Controller(){

};

Action Controller::steer_controller(const VehicleState& vs){

    constexpr float pi = 3.14159265359f;

    //use arctangent to calculate desired heading angle
    const float desiredHeading = std::atan2(vs.goaly - vs.y, vs.goalx - vs.x);

    //calculate shortest angle corrective between current heading and desired heading
    const float headingError = std::remainder(desiredHeading - vs.heading, 2.0f * pi);

    //PD controls
    constexpr float kp = 10.f;
    constexpr float kd = 2.f;
    constexpr float maxTorque = 10.f;

    Action action{};
    action.torque = std::clamp(
        kp * headingError - kd * vs.turnRate,
        -maxTorque, maxTorque
    );

    //create dynamic thrust
    constexpr float distGain = 5.f;
    constexpr float speedGain = 20.0f;
    constexpr float maxSpeed = 20.f;
    constexpr float maxThrust = 15.f;

    float distx = vs.x - vs.goalx;
    float disty = vs.y - vs.goaly;
    float dist = std::sqrt(distx*distx + disty*disty);

    //measure if pointing towards goal
    float alignment = std::max(0.0f, std::cos(headingError));

    //clamp speed at max and multiply by direction pointing
    float desiredSpeed = std::min(maxSpeed, dist*distGain) * alignment;

    //measure actual speed
    float actualSpeed = vs.vx * std::cos(vs.heading) 
                        + vs.vy * std::sin(vs.heading);

    //discrepency between desired and actual
    float speedError = desiredSpeed - actualSpeed;

    action.thrust = std::clamp(
        speedError * speedGain,
        -maxThrust,
        maxThrust
    );

    return action;
}