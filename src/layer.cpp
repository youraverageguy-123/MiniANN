#include "miniann/layer.hpp"

namespace miniann {

Layer::Layer(std::size_t numNeurons, std::size_t numInputs,
             std::function<std::unique_ptr<IActivation>()> makeAct,
             std::mt19937& rng) {
    neurons_.reserve(numNeurons);
    for (std::size_t i = 0; i < numNeurons; ++i)
        neurons_.emplace_back(numInputs, makeAct(), rng);
}

Vector Layer::forward(const Vector& inputs) {
    Vector out;
    out.reserve(neurons_.size());
    for (auto& n : neurons_) out.push_back(n.forward(inputs));
    return out;
}

} // namespace miniann
