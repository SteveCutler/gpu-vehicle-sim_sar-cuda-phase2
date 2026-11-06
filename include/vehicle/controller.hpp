#pragma once
#include "vehicle/vehicleBatch.hpp"
#include "vehicle/vehicleTypes.hpp"

class Controller
{
    public:
        Controller();

        Action steer_controller(const VehicleState& vs);
};

