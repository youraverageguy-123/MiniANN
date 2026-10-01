#include "miniann/dataset.hpp"
#include <stdexcept>

namespace miniann {

void Dataset::add(Vector input, Vector target) {
    inputs_.push_back(std::move(input));
    targets_.push_back(std::move(target));
}

Dataset Dataset::loadCSV(const std::string& /*path*/) {
    throw std::logic_error("Meeting 1: Dataset::loadCSV not implemented (Member 4 TODO)");
}

double accuracy(const std::vector<Vector>& /*preds*/,
                const std::vector<Vector>& /*targets*/) {
    throw std::logic_error("Meeting 1: accuracy() not implemented (Member 4 TODO)");
}

} // namespace miniann
