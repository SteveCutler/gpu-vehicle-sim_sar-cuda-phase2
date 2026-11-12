#pragma once
#include <string>

enum class PressureMode { Global, Tiled };
enum class CompactionMode { Atomic, PrefixScan };
enum class PlannerMode { RepeatedRollout, PrecomputedPaths };

struct ExperimentConfig {
    PressureMode pressure = PressureMode::Global;
    CompactionMode compaction = CompactionMode::Atomic;
    PlannerMode planner = PlannerMode::RepeatedRollout;
    bool use_graph = false;
};

ExperimentConfig loadExperimentConfig(const std::string& path);