#pragma once

struct DriftItem{
    float x;
    float y;

    DriftItem(){
        x = 0.0f;
        y = 0.0f;
    }

    DriftItem(float x_pos, float y_pos){
        x = x_pos;
        y = y_pos;
    }
};

struct PredictionParticle{
    std::size_t ID;
    DriftItem pos;
    float weight;
    float life_;

    PredictionParticle(std::size_t id){
        ID = id;
        pos.x = 0.f;
        pos.y = 0.f;
        weight = 0.f;
        life_ = 1.f;
    }

    PredictionParticle(float x_pos, float y_pos, std::size_t id){
        ID = id;
        pos.x = x_pos;
        pos.y = y_pos;
        weight = 0.f;
    }

    PredictionParticle(float x_pos, float y_pos, float life, std::size_t id){
        ID = id;
        pos.x = x_pos;
        pos.y = y_pos;
        weight = 0.f;
        life_ = life;
    }
};