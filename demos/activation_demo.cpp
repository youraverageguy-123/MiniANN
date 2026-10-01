// Meeting 1 demo: activation polymorphism works, nothing else is required.
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include "miniann/activation.hpp"

using namespace miniann;

int main() {
    std::vector<std::unique_ptr<IActivation>> acts;
    acts.push_back(std::make_unique<Sigmoid>());
    acts.push_back(std::make_unique<Tanh>());
    acts.push_back(std::make_unique<ReLU>());

    std::cout << "MiniANN Meeting 1: activation module\n";
    for (auto& a : acts) {
        // dynamic dispatch through IActivation* proves polymorphism
        const IActivation* base = a.get();
        std::cout << "  z=0.0 -> " << std::fixed << std::setprecision(4)
                  << base->activate(0.0)
                  << " (deriv " << base->derivative(0.0) << ")\n";
    }
    std::cout << "sigmoid(0)=" << Sigmoid().activate(0.0) << " (expect 0.5)\n";
    std::cout << "relu(-2)=" << ReLU().activate(-2.0) << " (expect 0)\n";
    std::cout << "STATUS: activation DONE; training/dataset/metrics are TODO stubs\n";
    return 0;
}
