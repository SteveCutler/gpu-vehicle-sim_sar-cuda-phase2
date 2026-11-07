#include "vehicle/controller.hpp"
#include <cuda_runtime.h>
#include "vehicle/controller_CUDA.hpp"

#include <cmath>
#include <algorithm>

__device__ ControllerCuda::ControllerCuda(){

};

__device__ Action ControllerCuda::steer_controller(const VehicleState& vs){

    constexpr float pi = 3.14159265359f;

    //use arctangent to calculate desired heading angle
    const float desiredHeading = atan2f(vs.goaly - vs.y, vs.goalx - vs.x);

    //calculate shortest angle corrective between current heading and desired heading
    const float headingError = remainderf(desiredHeading - vs.heading, 2.0f * pi);

    //PD controls
    constexpr float kp = 10.f;
    constexpr float kd = 2.f;
    constexpr float maxTorque = 10.f;

     Action action{};
    action.torque = fminf(maxTorque, fmaxf(
        -maxTorque, kp * headingError - kd * vs.turnRate
    ));

    //create dynamic thrust
    constexpr float distGain = 5.f;
    constexpr float speedGain = 20.0f;
    constexpr float maxSpeed = 20.f;
    constexpr float maxThrust = 15.f;

    float distx = vs.x - vs.goalx;
    float disty = vs.y - vs.goaly;
    float dist = sqrt(distx*distx + disty*disty);

    //measure if pointing towards goal
    float alignment = fmaxf(0.0f, cosf(headingError));

    //clamp speed at max and multiply by direction pointing
    float desiredSpeed = fminf(maxSpeed, dist*distGain) * alignment;

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