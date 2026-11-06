#pragma once
#include "vehicle/DriftItem.hpp"
#include <iostream>

class DriftDynamics
{

public:
    DriftDynamics() = default;
    
    //same update function used for driftItem and particlePredict, just with varying current fields
    void DriftUpdate(DriftItem& DI, std::pair<float,float> CurrentSample, float dt);


    ~DriftDynamics() = default;
};

