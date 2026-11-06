#include "vehicle/DriftDynamics.hpp"
#include "vehicle/DriftItem.hpp"

 void DriftDynamics::DriftUpdate(DriftItem& DI, std::pair<float,float> CurrentSample, float dt){
    DI.x += CurrentSample.first * dt;
    DI.y += CurrentSample.second * dt;

    return;
 };
 