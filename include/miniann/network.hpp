#pragma once
// Meeting 1 scope: NeuralNetwork = combination of Layers, predict() only.
// Training lives elsewhere (not implemented yet).
#include "miniann/layer.hpp"
#include <functional>
#include <memory>
#include <random>

namespace miniann {

class NeuralNetwork {
public:
    NeuralNetwork() = default;
    explicit NeuralNetwork(unsigned seed);

    // Member 1 (in progress): chain layers, validate dims. Throws invalid_argument.
    // NOTE: no backward()/zeroGradients() yet — those belong to the Training phase.
    void addLayer(std::size_t numNeurons, std::size_t numInputs,
                  std::function<std::unique_ptr<IActivation>()> makeAct);

    Vector predict(const Vector& input);

    std::size_t numLayers() const { return layers_.size(); }
    std::vector<Layer>& layers() { return layers_; }
    const std::vector<Layer>& layers() const { return layers_; }

private:
    std::vector<Layer> layers_;
    std::mt19937 rng_{std::random_device{}()};
};

inline std::function<std::unique_ptr<IActivation>()> useSigmoid() {
    return []() -> std::unique_ptr<IActivation> { return std::make_unique<Sigmoid>(); };
}
inline std::function<std::unique_ptr<IActivation>()> useTanh() {
    return []() -> std::unique_ptr<IActivation> { return std::make_unique<Tanh>(); };
}
inline std::function<std::unique_ptr<IActivation>()> useReLU() {
    return []() -> std::unique_ptr<IActivation> { return std::make_unique<ReLU>(); };
}

} // namespace miniann
