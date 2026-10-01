#pragma once
// Meeting 1 scope: Training System is PLANNED, not implemented.
// This header only freezes the intended interface for Member 3.
// Any call throws std::logic_error("not implemented in Meeting 1").
#include "miniann/types.hpp"

namespace miniann {

class ILoss {
public:
    virtual ~ILoss() = default;
    virtual double compute(const Vector& predicted, const Vector& target) const = 0;
};

class MSELoss : public ILoss {
public:
    double compute(const Vector& predicted, const Vector& target) const override;
};

class IOptimizer {
public:
    virtual ~IOptimizer() = default;
    // TODO(Meeting 2+): SGD::step
    virtual void step() = 0;
};

class SGD : public IOptimizer {
public:
    explicit SGD(double lr);
    void step() override; // throws until Training phase
private:
    double lr_ = 0.01;
};

} // namespace miniann
