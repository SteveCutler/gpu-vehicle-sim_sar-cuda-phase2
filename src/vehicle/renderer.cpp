#include "vehicle/renderer.hpp"
#include <vector>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <algorithm>
#include "vehicle/environment.hpp"
#include "vehicle/DriftItem.hpp"
#include "vehicle/ProbabilityMap.hpp"

Renderer::Renderer(std::size_t width, std::size_t height, bool arrow_viz, std::size_t cell_size):
m_width(width),
m_height(height),
m_fluid_texture(sf::Vector2u(width, height)),
m_scale(sf::Vector2f(1.0f, 1.0f)),
m_sprite(m_fluid_texture),
m_arrow_viz(arrow_viz),
m_arrows(sf::PrimitiveType::Lines, width/16 * height/16 * 6),
m_p_grid(sf::PrimitiveType::Triangles, width/cell_size * height/cell_size * 6),
m_window(sf::VideoMode({m_width, m_height}), "Search&Rescue Sim")
{
    //Render logic initialization
    m_sprite.setScale(m_scale);
    m_window.setFramerateLimit(60);
             //create window logic

    //TO DO 
    //display performance time here

    //initialize probabilty map triangle grid

    std::size_t cols = std::floor(width / cell_size);
    std::size_t rows = std::floor(height / cell_size);

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int i = (row * cols + col) * 6;

            float x = col * cell_size;
            float y = row * cell_size;

            // First triangle
            m_p_grid[i + 0].position = {x, y};
            m_p_grid[i + 1].position = {x + cell_size, y};
            m_p_grid[i + 2].position = {x, y + cell_size};

            // Second triangle
            m_p_grid[i + 3].position = {x + cell_size, y};
            m_p_grid[i + 4].position = {x + cell_size, y + cell_size};
            m_p_grid[i + 5].position = {x, y + cell_size};
        }
    }

};

bool Renderer::draw(const vehicleBatch& vehicles, float dt, std::size_t curr_step, const environment& env, const DriftItem& target, const ProbabilityMap& p_map){

    //check if window has been closed:
    while (auto event = m_window.pollEvent()) {
        if (event->is<sf::Event::Closed>()) {
            m_window.close();
            std::cout << "window closed" << std::endl;
        }
    }

    if (!m_window.isOpen()) return false;

    
    //vehicle count
    const std::size_t count = vehicles.getSize();

    //create vehicle shape
    sf::ConvexShape vehicleShape(3);
    vehicleShape.setOrigin({0.f, 0.f});
    vehicleShape.setPoint(0, {10.f, 0.f});   
    vehicleShape.setPoint(1, {-7.f, -6.f});
    vehicleShape.setPoint(2, {-7.f,  6.f});
    vehicleShape.setFillColor(sf::Color::Cyan);


    //create particle shape
    sf::CircleShape particleShape(1.f);
    particleShape.setOrigin({4.f, 4.f});
    particleShape.setFillColor(sf::Color::Green);

    //create target shape
    sf::ConvexShape targetShape(10);
    targetShape.setPoint(0, {0.f, -5.f});
    targetShape.setPoint(1, {1.18f, -1.62f});
    targetShape.setPoint(2, {4.76f, -1.55f});
    targetShape.setPoint(3, {1.9f, 0.62f});
    targetShape.setPoint(4, {2.94f, 4.05f});
    targetShape.setPoint(5, {0.f, 2.f});
    targetShape.setPoint(6, {-2.94f, 4.05f});
    targetShape.setPoint(7, {-1.9f, 0.62f});
    targetShape.setPoint(8, {-4.76f, -1.55f});
    targetShape.setPoint(9, {-1.18f, -1.62f});
    targetShape.setFillColor(sf::Color::Red);

    //create goal Shape
    sf::ConvexShape goalShape(10);
    goalShape = targetShape;
    goalShape.setFillColor(sf::Color::Blue);


    //clear previous frame    
    m_window.clear(sf::Color::Black);

    // draw velocity arrow samples

    if(m_arrow_viz){
        //extract u and v velocity directions
       // auto [u, v] = env.getDisturbance(x, y, dt*curr_step);
        const auto& u_velocity = env.fluid_grid.u_velocity();
        const auto& v_velocity = env.fluid_grid.v_velocity();
        std::size_t index=0;
        for (std::size_t y = 0; y<m_height; y+=16){
            std::size_t row = y*m_width;
            
            for(std::size_t x =0; x<m_width; x+=16){
                std::size_t i = row+x;
                

                //increase length for display
              //  u *= 10;
              //  v *= 10;
                
                //create arrow points
                m_arrows[index].position = sf::Vector2f(static_cast<float>(x),static_cast<float>(y));
                m_arrows[index+1].position = sf::Vector2f(static_cast<float>(x+u_velocity[i]),static_cast<float>(y+v_velocity[i]));
                
                m_arrows[index+2].position = sf::Vector2f(static_cast<float>(x+u_velocity[i]),static_cast<float>(y+v_velocity[i]));
                m_arrows[index+3].position = sf::Vector2f(static_cast<float>(x+u_velocity[i] - u_velocity[i]*.25 + v_velocity[i]*.25),static_cast<float>(y+v_velocity[i] - v_velocity[i]*.25-u_velocity[i]*.25));
                
                m_arrows[index+4].position = sf::Vector2f(static_cast<float>(x+u_velocity[i]),static_cast<float>(y+v_velocity[i]));
                m_arrows[index+5].position = sf::Vector2f(static_cast<float>(x+u_velocity[i] - u_velocity[i]*.25 - v_velocity[i]*.25),static_cast<float>(y+v_velocity[i] - v_velocity[i]*.25+u_velocity[i]*.25));
                
                
                index += 6;
            }

        }
    }

    //Load fluid data into texture
    const auto& pixels = env.get_pixels();
    // Load pixel RGBA data into texture
    m_fluid_texture.update(pixels.data());
    m_window.draw(m_sprite);


    //clear probability grid to black
    for(std::size_t i = 0; i < m_p_grid.getVertexCount(); i++){
        m_p_grid[i].color = sf::Color::Transparent;
    }

    
    if(!p_map.occupied_cells.empty()){
        for(std::size_t i = 0; i < p_map.occupied_cells.size(); i++){
            //cell with particles in it
            std::size_t cell_num = p_map.occupied_cells[i];
            float intensity = (p_map.map_[cell_num]/p_map.particles_.size()) * 255.0;
            const float fraction = p_map.map_[cell_num] / p_map.particles_.size();
            const auto alpha = static_cast<std::uint8_t>(255.f * fraction);
            
            std::size_t start_vert = cell_num * 6;
            
            m_p_grid[start_vert + 0].color = sf::Color(0, 255, 0, alpha);
            m_p_grid[start_vert + 1].color = sf::Color(0, 255, 0, alpha);
            m_p_grid[start_vert + 2].color = sf::Color(0, 255, 0, alpha);
            
            // Second triangle
            m_p_grid[start_vert + 3].color = sf::Color(0, 255, 0, alpha);
            m_p_grid[start_vert + 4].color = sf::Color(0, 255, 0, alpha);
            m_p_grid[start_vert + 5].color = sf::Color(0, 255, 0, alpha);
        }
        
    }
    //draw probability map
    m_window.draw(m_p_grid);
    
    
    //render arrows
    if(m_arrow_viz){
        m_window.draw(m_arrows);
    }

    //drawing probability particles
    // for(size_t p = 0; p < p_map.particles_.size(); p++){
    //     particleShape.setPosition({p_map.particles_[p].pos.x, p_map.particles_[p].pos.y});
    //     m_window.draw(particleShape);
    // }

        // Load and render vehicle data
    for(int i = 0; i<count; i++){
        VehicleState s = vehicles.load(i);

        vehicleShape.setPosition({s.x, m_height - s.y});
        vehicleShape.setRotation(sf::radians(-s.heading));
        vehicleShape.setScale(sf::Vector2f(0.7f,0.7f));
        
        m_window.draw(vehicleShape);

        //set and draw goal shape
        goalShape.setPosition({s.goalx, m_height - s.goaly});
        m_window.draw(goalShape);

            //create sensor shape
        sf::CircleShape sensorShape(s.sensor_rad);
        sensorShape.setOrigin({s.sensor_rad, s.sensor_rad});
        sensorShape.setScale({.5f,.5f});
        sensorShape.setFillColor(sf::Color::Transparent);
        sensorShape.setOutlineThickness(0.6f);
        sensorShape.setOutlineColor(sf::Color::White);

        
        sensorShape.setPosition({s.x, m_height - s.y});
        sensorShape.setScale(sf::Vector2f(2.5f,2.5f));
        m_window.draw(sensorShape);
    }
    
    
    //draw target
    targetShape.setPosition({target.x,target.y});
    m_window.draw(targetShape);

    m_window.display();



    return true;


};