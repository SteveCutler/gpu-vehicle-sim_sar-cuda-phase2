#include "ExperimentConfig.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");

    if (first == std::string::npos) {
        return "";
    }

    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

} // namespace

ExperimentConfig loadExperimentConfig(const std::string& path) {
    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    ExperimentConfig config;
    std::unordered_set<std::string> seen_keys;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(file, line)) {
        ++line_number;

        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }

        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error(
                "Expected key=value on line " + std::to_string(line_number)
            );
        }

        const std::string key = trim(line.substr(0, equals));
        const std::string value = trim(line.substr(equals + 1));

        if (!seen_keys.insert(key).second) {
            throw std::runtime_error("Duplicate config key: " + key);
        }

        if (key == "pressure") {
            if (value == "global") {
                config.pressure = PressureMode::Global;
            } else if (value == "tiled") {
                config.pressure = PressureMode::Tiled;
            } else {
                throw std::runtime_error("Unknown pressure mode: " + value);
            }
        } else if (key == "compaction") {
            if (value == "atomic") {
                config.compaction = CompactionMode::Atomic;
            } else if (value == "prefix_scan") {
                config.compaction = CompactionMode::PrefixScan;
            } else {
                throw std::runtime_error("Unknown compaction mode: " + value);
            }
        } else if (key == "planner") {
            if (value == "repeated") {
                config.planner = PlannerMode::RepeatedRollout;
            } else if (value == "precomputed") {
                config.planner = PlannerMode::PrecomputedPaths;
            } else {
                throw std::runtime_error("Unknown planner mode: " + value);
            }
        } else if (key == "use_graph") {
            if (value == "true") {
                config.use_graph = true;
            } else if (value == "false") {
                config.use_graph = false;
            } else {
                throw std::runtime_error("use_graph must be true or false");
            }
        } else {
            throw std::runtime_error("Unknown config key: " + key);
        }
    }

    return config;
}