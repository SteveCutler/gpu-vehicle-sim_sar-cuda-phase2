#include "fluid/Slider.hpp"
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

sf::Font font;

bool loaded = font.openFromFile(FLUID_FONT_PATH);




Slider::Slider(std::string title, sf::Vector2f position, float min_value, float max_value, float* target):
slider_title(font){
    this->target=target;
    this->min_value=min_value;
    this->max_value=max_value;
    
    slider_title.setCharacterSize(12);
    slider_title.setFillColor(sf::Color::White);
    slider_title.setPosition({position.x, position.y-17.5f});
    slider_title.setString(title);
    
    track.setFillColor({100, 100, 100});
    track.setPosition(position);
    track.setSize(sf::Vector2f(120.f,5.f));
    
    float amount =std::clamp(((*target - min_value)/(max_value-min_value)),0.f,1.f);
    
    float knob_pos_x = track.getPosition().x + track.getSize().x*amount;
    float knob_pos_y = track.getPosition().y - track.getSize().y/2.f;

    knob.setPosition({knob_pos_x,knob_pos_y});
    knob.setFillColor(sf::Color::Red);
    knob.setScale({4.f,4.f});
    knob.setRadius(1.f);
}

void Slider::updateFromMouse(sf::Vector2f pos){
    float percentage = (pos.x-track.getPosition().x)/track.getSize().x; 
    float amount = min_value + (max_value-min_value)*percentage;
    *target = std::clamp(amount,min_value, max_value);
    float knobPos = std::clamp((track.getPosition().x + track.getSize().x*percentage), track.getPosition().x,(track.getPosition().x+track.getSize().x));

        knob.setPosition({
            knobPos,
            knob.getPosition().y
        }
    );
}

bool Slider::contains(sf::Vector2f pos){
    
    if(track.getGlobalBounds().contains(pos) or knob.getGlobalBounds().contains(pos)){
        return true;

    }
    return false;
}

//draw function
void Slider::draw(sf::RenderWindow& window){
    window.draw(track);
    window.draw(knob);
    window.draw(slider_title);
}

