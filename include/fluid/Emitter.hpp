#pragma once

#include <SFML/System/Vector3.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Graphics/Color.hpp>

class Emitter
{

public:
Emitter(sf::Vector2f pos, float rad, sf::Vector3f clr);

sf::Vector2f m_pos;
float m_rad;
sf::Vector3f m_clr;
};



