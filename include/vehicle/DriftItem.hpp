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
    DriftItem pos;
    float weight;
    float life_;

    PredictionParticle(){
        pos.x = 0.f;
        pos.y = 0.f;
        weight = 0.f;
        life_ = 1.f;
    }

    PredictionParticle(float x_pos, float y_pos){
        pos.x = x_pos;
        pos.y = y_pos;
        weight = 0.f;
    }

    PredictionParticle(float x_pos, float y_pos, float life){
        pos.x = x_pos;
        pos.y = y_pos;
        weight = 0.f;
        life_ = life;
    }
};