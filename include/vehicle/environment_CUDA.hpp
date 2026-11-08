#pragma once
#include <vector>
#include "fluid/FluidGrid_CUDA.hpp"
#include "fluid/Emitter.hpp"

class environment_CUDA
{
public: 

std::size_t m_width;
std::size_t m_height;
FluidGrid_CUDA fluid_grid;
std::vector<Emitter*> emitters;
std::size_t seed;

private:
std::vector<float> velFieldx, velFieldy;

public:
environment_CUDA(std::size_t w, std::size_t h, std::vector<Emitter*>& emitters, std::size_t seed);

void update_fluid(float dt);

const std::vector<std::uint8_t> & get_pixels() const;

std::pair<float,float> getDisturbance(float x, float y, float time) const;

};

