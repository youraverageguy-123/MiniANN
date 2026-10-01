#include "miniann/activation.hpp"

namespace miniann {

double Sigmoid::activate(double z) const {
    return 1.0 / (1.0 + std::exp(-z));
}
double Sigmoid::derivative(double z) const {
    double a = activate(z);
    return a * (1.0 - a);
}

double Tanh::activate(double z) const {
    return std::tanh(z);
}
double Tanh::derivative(double z) const {
    double a = std::tanh(z);
    return 1.0 - a * a;
}

double ReLU::activate(double z) const {
    return z > 0.0 ? z : 0.0;
}
double ReLU::derivative(double z) const {
    return z > 0.0 ? 1.0 : 0.0;
}

} // namespace miniann
