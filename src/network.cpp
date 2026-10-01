#include "miniann/network.hpp"
#include <stdexcept>

namespace miniann {

NeuralNetwork::NeuralNetwork(unsigned seed) : rng_(seed) {}

void NeuralNetwork::addLayer(std::size_t numNeurons, std::size_t numInputs,
                             std::function<std::unique_ptr<IActivation>()> makeAct) {
    if (!layers_.empty() && layers_.back().size() != numInputs)
        throw std::invalid_argument("addLayer: input size must match previous layer size");
    layers_.emplace_back(numNeurons, numInputs, std::move(makeAct), rng_);
}

Vector NeuralNetwork::predict(const Vector& input) {
    if (layers_.empty()) throw std::runtime_error("predict: network has no layers");
    Vector out = input;
    for (auto& l : layers_) out = l.forward(out);
    return out;
}

} // namespace miniann
