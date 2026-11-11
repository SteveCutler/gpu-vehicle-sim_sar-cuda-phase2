#pragma once
#include <cstddef>

#include <vector>

struct RoutePoint {
    float x;
    float y;
};

struct VehiclePreview {
    std::vector<std::vector<RoutePoint>> candidates; // one path per waypoint
    std::size_t selected = 0;                        // winning candidate
};

struct PlannerPreview {
    std::vector<VehiclePreview> vehicles;             // one per vehicle
    std::vector<std::vector<RoutePoint>> particles;   // sampled particle paths
};