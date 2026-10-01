#pragma once
// Meeting 1 scope: activation module is DONE.
// Matches the PDF interface exactly: activate(z) + derivative(z), no factory yet.
#include <cmath>

namespace miniann {

class IActivation {
public:
    virtual ~IActivation() = default;
    virtual double activate(double z) const = 0;
    virtual double derivative(double z) const = 0;
};

class Sigmoid : public IActivation {
public:
    double activate(double z) const override;
    double derivative(double z) const override;
};

class Tanh : public IActivation {
public:
    double activate(double z) const override;
    double derivative(double z) const override;
};

class ReLU : public IActivation {
public:
    double activate(double z) const override;
    double derivative(double z) const override;
};

} // namespace miniann
