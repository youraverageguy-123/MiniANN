#pragma once
// Meeting 1 scope: Layer = collection of neurons, forward only.
#include "miniann/neuron.hpp"
#include <functional>

namespace miniann {

class Layer {
public:
    Layer(std::size_t numNeurons, std::size_t numInputs,
          std::function<std::unique_ptr<IActivation>()> makeAct,
          std::mt19937& rng);

    Vector forward(const Vector& inputs);

    std::size_t size() const { return neurons_.size(); }
    std::vector<Neuron>& neurons() { return neurons_; }
    const std::vector<Neuron>& neurons() const { return neurons_; }

private:
    std::vector<Neuron> neurons_;
};

} // namespace miniann
