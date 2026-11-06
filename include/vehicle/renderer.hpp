#pragma once
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include "vehicle/vehicleTypes.hpp"
#include "vehicle/vehicleBatch.hpp"
#include "vehicle/DriftItem.hpp"
#include "vehicle/ProbabilityMap.hpp"

class Renderer
{
private:
    unsigned int m_width;
    unsigned int m_height;
    sf::Texture m_fluid_texture;
    sf::Vector2f m_scale;
    sf::Sprite m_sprite;
    bool m_arrow_viz;
    sf::VertexArray m_arrows;
    sf::VertexArray m_p_grid;
    sf::RenderWindow m_window;

public:
    Renderer(std::size_t width, std::size_t height, bool arrow_viz, std::size_t cell_size);
    bool draw(const vehicleBatch& vehicles, float dt, std::size_t curr_step, const environment& env, const DriftItem& target, const ProbabilityMap& p_map);
    ~Renderer() = default;
};


