// Meeting 1 test: activation values only. No gradient check / serialization yet.
#include <iostream>
#include <cassert>
#include <cmath>
#include "miniann/activation.hpp"

using namespace miniann;

int main() {
    Sigmoid s;
    Tanh t;
    ReLU r;
    assert(std::abs(s.activate(0.0) - 0.5) < 1e-12);
    assert(std::abs(s.derivative(0.0) - 0.25) < 1e-12);
    assert(std::abs(t.activate(0.0)) < 1e-12);
    assert(r.activate(-2.0) == 0.0);
    assert(r.derivative(-2.0) == 0.0);
    assert(r.derivative(3.0) == 1.0);

    // polymorphism check
    const IActivation* p = &s;
    assert(std::abs(p->activate(0.0) - 0.5) < 1e-12);

    std::cout << "MEETING1 ACTIVATION TESTS PASS\n";
    std::cout << "TODO (later phases): gradient check, serializer round-trip, XOR learning\n";
    return 0;
}
