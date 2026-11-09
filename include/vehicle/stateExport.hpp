#pragma once
#include "vehicle/vehicleBatch.hpp"
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include "vehicle/DriftItem.hpp"
#include <vector>

inline void exportStates(
    const vehicleBatch& vehicles,
    const std::string& path)
{
    std::ofstream out(path);

    if (!out) {
        throw std::runtime_error("Cannot open output file: " + path);
    }

    out << std::setprecision(
        std::numeric_limits<float>::max_digits10
    );

    out << "id,x,y,vx,vy,heading,turnRate,goalx,goaly\n";

    for (std::size_t i = 0; i < vehicles.getSize(); ++i) {
        const VehicleState vs = vehicles.load(i);

        out << i << ','
            << vs.x << ',' << vs.y << ','
            << vs.vx << ',' << vs.vy << ','
            << vs.heading << ',' << vs.turnRate << ','
            << vs.goalx << ',' << vs.goaly << '\n';
    }

    out.close();

    if (!out) {
        throw std::runtime_error("Failed to write states: " + path);
    }
}

inline void exportParticles(
    const std::vector<PredictionParticle>& particles,
    const std::string& path)
{
    std::ofstream out(path);

    if (!out) {
        throw std::runtime_error("Cannot open output file: " + path);
    }

    out << std::setprecision(
        std::numeric_limits<float>::max_digits10);

    out << "index,x,y,weight,life\n";

    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto& particle = particles[i];

        out << i << ','
            << particle.pos.x << ','
            << particle.pos.y << ','
            << particle.weight << ','
            << particle.life_ << '\n';
    }

    out.close();

    if (!out) {
        throw std::runtime_error("Failed to write particles: " + path);
    }
}