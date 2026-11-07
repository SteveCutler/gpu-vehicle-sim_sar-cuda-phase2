cmake_minimum_required(VERSION 3.18)
project(gpu_vehicle_sim LANGUAGES CXX)

option(ENABLE_RENDERER "Build the SFML renderer" OFF)
option(ENABLE_CUDA "Build CUDA support" OFF)

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)

# Fluid computation.
add_library(fluid_core STATIC
    src/fluid/FluidGrid.cpp
    src/fluid/Emitter.cpp
)

target_link_libraries(fluid_core 
PUBLIC
SFML::System
PRIVATE
    SFML::Graphics
    SFML::Window

)



target_include_directories(fluid_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(fluid_core PUBLIC cxx_std_17)

# Main application.
add_executable(vehicle_sim
    src/main.cpp
    src/vehicle/vehicleBatch.cpp
    src/vehicle/environment.cpp
    src/vehicle/dynamics.cpp
    src/vehicle/controller.cpp
    src/vehicle/DriftDynamics.cpp
    src/vehicle/ProbabilityMap.cpp
    src/vehicle/SearchPlanner.cpp
)

target_link_libraries(vehicle_sim 
PRIVATE 
fluid_core

)

if(ENABLE_CUDA)

    target_sources(fluid_core PRIVATE
        src/fluid/FluidGrid.cu

    )
    target_sources(vehicle_sim PRIVATE
        src/vehicle/ProbabilityMap.cu
        src/vehicle/SearchPlanner.cu
        src/CUDA_simulation.cu
    )

    target_link_libraries(fluid_core PRIVATE CUDA::cudart)
    target_link_libraries(vehicle_sim PRIVATE CUDA::cudart)

    # Allow #ifdef ENABLE_CUDA in each target's source files.
    target_compile_definitions(fluid_core PRIVATE ENABLE_CUDA)
    target_compile_definitions(vehicle_sim PRIVATE ENABLE_CUDA)

    set_target_properties(fluid_core vehicle_sim PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()


# Optional rendering and UI.
if(ENABLE_RENDERER)

    target_sources(vehicle_sim PRIVATE
        src/vehicle/renderer.cpp
        src/fluid/FluidGrid.cpp
        src/fluid/Slider.cpp
    )

    target_link_libraries(vehicle_sim PRIVATE
        SFML::Graphics
        SFML::Window
        SFML::System
    )

    target_compile_definitions(vehicle_sim PRIVATE
        ENABLE_RENDERER
        "FLUID_FONT_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/assets/fonts/RobotoMono.ttf\""
    )
endif()