#pragma once
#include "vehicle/vehicleTypes.hpp"
#include <cuda_runtime.h>
#include "vehicle/environment.hpp"


class DynamicsCuda
{

public:
    __device__ DynamicsCuda();

    __device__ VehicleState step_update(const VehicleState& vs, const Action& action, cuda::std::pair<float,float> current, const float dt);
};

