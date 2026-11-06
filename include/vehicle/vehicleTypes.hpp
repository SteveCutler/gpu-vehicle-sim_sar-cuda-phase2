#pragma once

struct VehicleState{
    float x, y;
    float vx, vy;
    float heading, turnRate;
    float goalx, goaly;
    float sensor_rad;
};

struct Action{
    float thrust, torque;
};