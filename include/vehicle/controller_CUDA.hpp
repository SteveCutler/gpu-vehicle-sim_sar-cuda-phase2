#pragma once
#include <cuda_runtime.h>
#include "vehicle/vehicleTypes.hpp"

class ControllerCuda
{
    public:
    __device__ ControllerCuda();

    __device__ Action steer_controller(const VehicleState& vs);
};
