#include "miniann/neuron.hpp"

namespace miniann {

Neuron::Neuron(std::size_t numInputs, std::unique_ptr<IActivation> act, std::mt19937& rng)
    : activation_(std::move(act)) {
    // Meeting 1: simple uniform init only. Xavier/He come later.
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    weights_.resize(numInputs);
    for (auto& w : weights_) w = d(rng);
    bias_ = d(rng) * 0.5;
}

double Neuron::forward(const Vector& inputs) {
    double z = bias_;
    for (std::size_t i = 0; i < weights_.size() && i < inputs.size(); ++i)
        z += weights_[i] * inputs[i];
    return activation_->activate(z);
}

} // namespace miniann
