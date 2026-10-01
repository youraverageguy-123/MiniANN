// Meeting 1 demo: untrained forward pass only. Proves composition
// NeuralNetwork -> Layer -> Neuron -> IActivation, but NO learning happens.
// XOR outputs below are essentially random — training is a later phase.
#include <iostream>
#include <iomanip>
#include "miniann/network.hpp"
#include "miniann/training.hpp"

using namespace miniann;

int main() {
    NeuralNetwork net(42);
    net.addLayer(2, 2, useTanh());
    net.addLayer(1, 2, useSigmoid());

    const double xs[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    MSELoss loss;
    std::cout << "MiniANN Meeting 1: forward-only XOR (UNTRAINED)\n";
    for (auto& x : xs) {
        Vector out = net.predict({x[0], x[1]});
        double l = loss.compute(out, {0.0}); // loss value only; no gradient/backprop yet
        std::cout << "  [" << x[0] << "," << x[1] << "] -> "
                  << std::fixed << std::setprecision(4) << out[0]
                  << " (mse-vs-0 " << l << ")\n";
    }
    try {
        SGD opt(0.5);
        opt.step();
    } catch (const std::exception& e) {
        std::cout << "training stub correctly throws: " << e.what() << "\n";
    }
    std::cout << "STATUS: forward works; backward/optimizer/trainer NOT implemented\n";
    return 0;
}
