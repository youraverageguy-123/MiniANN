#pragma once
// Meeting 1 scope: Neuron stores weights/bias and does forward only.
// No backward(), no gradient accumulation, no Xavier/He yet (uniform init).
#include "miniann/types.hpp"
#include "miniann/activation.hpp"
#include <memory>
#include <random>
#include <cstddef>

namespace miniann {

class Neuron {
public:
    Neuron(std::size_t numInputs, std::unique_ptr<IActivation> act, std::mt19937& rng);

    // z = w.x + b, output = activation(z)
    double forward(const Vector& inputs);

    const Vector& weights() const { return weights_; }
    double bias() const { return bias_; }
    const IActivation& activation() const { return *activation_; }

private:
    Vector weights_;
    double bias_ = 0.0;
    std::unique_ptr<IActivation> activation_;
};

} // namespace miniann
