#pragma once
// Meeting 1 scope: Dataset + Evaluation are PLANNED, not implemented.
// Member 4 owns these in a later phase. Calls throw logic_error.
#include "miniann/types.hpp"
#include <string>
#include <vector>

namespace miniann {

class Dataset {
public:
    void add(Vector input, Vector target);
    std::size_t size() const { return inputs_.size(); }
    const Vector& input(std::size_t i) const { return inputs_.at(i); }
    const Vector& target(std::size_t i) const { return targets_.at(i); }

    // TODO(Meeting 3+): loadCSV(), split(), validate(), makeXorGate()...
    static Dataset loadCSV(const std::string& path); // throws until implemented
private:
    std::vector<Vector> inputs_;
    std::vector<Vector> targets_;
};

double accuracy(const std::vector<Vector>& preds,
                const std::vector<Vector>& targets); // throws until implemented

} // namespace miniann
