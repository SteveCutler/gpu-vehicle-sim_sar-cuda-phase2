#pragma once


namespace constants {
    //controller
    inline constexpr float kp = 10.f;
    inline constexpr float kd = 2.f;
    inline constexpr float maxTorque = 10.f;

    inline constexpr float distGain = 5.f;
    inline constexpr float speedGain = 20.0f;
    inline constexpr float maxSpeed = 20.f;
    inline constexpr float maxThrust = 15.f;

    //dyanmics
    inline constexpr float mass = 1.f;
    inline constexpr float momentOfInertia = 1.f;
    inline constexpr float linearDrag = 0.2f;
    inline constexpr float turnDrag = 0.35f;
    inline constexpr float pi = 3.14159265359f;

}
