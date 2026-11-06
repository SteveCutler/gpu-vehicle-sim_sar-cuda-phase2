#pragma once
#include "vehicle/vehicleTypes.hpp"
#include "vehicle/environment.hpp"


class Dynamics
{

public:
    Dynamics();

    VehicleState step_update(const VehicleState& vs, const Action& action, const environment& env, const float dt, std::size_t steps);
};

