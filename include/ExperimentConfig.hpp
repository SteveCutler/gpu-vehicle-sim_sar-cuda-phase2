#pragma once
#include <string>

enum class PressureMode {
    Global,
    Tiled,
    Temporal
};

enum class CompactionMode {
    Atomic,
    PrefixGlobal,
    PrefixShared
};

enum class PlannerMode {
    RepeatedRollout,
    PrecomputedPaths
};

enum class HitMode {
    Bytes,
    Packed,
    Fused
};

struct ExperimentConfig {
    PressureMode pressure = PressureMode::Global;
    CompactionMode compaction = CompactionMode::Atomic;
    PlannerMode planner = PlannerMode::RepeatedRollout;
    HitMode hits = HitMode::Bytes;
    bool use_graph = false;
};

ExperimentConfig loadExperimentConfig(const std::string& path);