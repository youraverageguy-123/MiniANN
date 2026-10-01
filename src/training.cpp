#include "miniann/training.hpp"
#include <stdexcept>

namespace miniann {

double MSELoss::compute(const Vector& predicted, const Vector& target) const {
    if (predicted.size() != target.size())
        throw std::invalid_argument("MSELoss: size mismatch");
    double s = 0.0;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        double d = predicted[i] - target[i];
        s += d * d;
    }
    return predicted.empty() ? 0.0 : s / double(predicted.size());
}

SGD::SGD(double lr) : lr_(lr) {}

void SGD::step() {
    throw std::logic_error("Meeting 1: SGD::step not implemented (Training phase TODO, Member 3)");
}

} // namespace miniann
