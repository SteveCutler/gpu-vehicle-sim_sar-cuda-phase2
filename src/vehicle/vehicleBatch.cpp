#include "vehicle/vehicleBatch.hpp"
#include <random>


vehicleBatch::vehicleBatch(std::size_t N, environment env, float goal_x, float goal_y ):
    m_x(N),
    m_y(N),
    m_vx(N),
    m_vy(N),
    m_heading(N),
    m_turnRate(N),
    m_goalx(N),
    m_goaly(N),
    m_size(N),
    m_sensor_rad(N),
    init_goalx(goal_x),
    init_goaly(goal_y)
{

    //RNG setup 
    std::mt19937 rng(25);

    constexpr float margin = 10.f;
    constexpr float pi = 3.14159265359f;

    std::uniform_real_distribution<float> randomX(
        margin, static_cast<float>(env.m_width) - margin);

    std::uniform_real_distribution<float> randomY(
        margin, static_cast<float>(env.m_height) - static_cast<float>(env.m_height)*.5);

    std::uniform_real_distribution<float> randomHeading(-pi, pi);


    for(int x = 0; x < N; x++){
      //initializing vehicles
        m_x[x] = randomX(rng);
        m_y[x] = randomY(rng);

        m_vx[x] = 0.f;
        m_vy[x] = 0.f;

        m_heading[x] = randomHeading(rng);
        m_turnRate[x] = 0.f;

        m_goalx[x] = init_goalx;
        m_goaly[x] = init_goaly;
        
        m_sensor_rad[x] = 10.f;
        
    }

};

VehicleState vehicleBatch::load(std::size_t i) const{

    return {
        m_x[i], 
        m_y[i], 
        m_vx[i], 
        m_vy[i], 
        m_heading[i], 
        m_turnRate[i],
        m_goalx[i],
        m_goaly[i],
        m_sensor_rad[i]
    };

};

void vehicleBatch::set(std::size_t i, const VehicleState& newState){
    m_x[i] = newState.x;
    m_y[i] = newState.y;
    m_vx[i] = newState.vx;
    m_vy[i] = newState.vy;
    m_heading[i] = newState.heading;
    m_turnRate[i] = newState.turnRate;
    m_goalx[i] = newState.goalx;
    m_goaly[i] = newState.goaly;
};

std::size_t vehicleBatch::getSize() const{
    return m_size;
};

bool vehicleBatch::scan(VehicleState& vs, ProbabilityMap& p_map, DriftItem& target){

    //check if target in censor
    float td_x = target.x - vs.x;
    float td_y = (target.y) - (p_map.height_ - vs.y);
    float target_dist = std::sqrt(td_x*td_x + td_y*td_y);

    //check if goal in censor
    float tg_x = vs.goalx - vs.x;
    float tg_y = (p_map.height_ - vs.goaly) - (p_map.height_ - vs.y);
    float goal_dist = std::sqrt(tg_x*tg_x + tg_y*tg_y);


    if(target_dist <= vs.sensor_rad*2){
        std::cout << "Found target!" << std::endl;
        return true;
    }


    if(goal_dist < vs.sensor_rad*2){

        p_map.updateGoal(vs);
        
    }

    p_map.particleScan(vs);

    return false;

}

/*
TO DO

write function to assign random x y vx vy heading turnrate values on initialization

every vehicle same goal for now

write update function that takes in action and vehicle state structs
*/