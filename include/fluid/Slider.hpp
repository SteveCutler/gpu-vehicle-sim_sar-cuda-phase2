#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>


struct Slider{
    sf::RectangleShape track;
    sf::CircleShape knob;
    sf::Text slider_title;

    float* target;
    float min_value;
    float max_value;
    bool dragging;



    Slider(std::string title, sf::Vector2f position, float min_value, float max_value, float* target);

    void updateFromMouse(sf::Vector2f pos);

    bool contains(sf::Vector2f pos);

    //draw function
    void draw(sf::RenderWindow& window);

};