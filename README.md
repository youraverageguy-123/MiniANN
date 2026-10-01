# MiniANN: C++ Artificial Neural Network Library

A lightweight Artificial Neural Network framework implemented from scratch in modern C++.
Built around architectural clarity, strict encapsulation, and clean Object-Oriented Design.

Problem Statement 5 · C++17 · no external ML frameworks · 4-member team.

## Overview

An Artificial Neural Network consists of interconnected computational units called neurons.
Each neuron combines inputs with learned weights, adds a bias, and applies an activation
function:

```text
z = w · x + b
output = activation(z)
```

MiniANN maps each of these mathematical concepts to a class or interface. The model follows
a composition-based design:

```text
Neural Network
    |
  Layers
    |
  Neurons
    |
Activation Functions

Training Module
Dataset Module
Evaluation Module
```

A core architectural decision: **the network does not train itself**. Training logic lives in
a separate module, so learning strategies can change without modifying the model classes.

OOP concepts applied throughout: abstraction, inheritance, polymorphism, encapsulation,
and composition.

## Features

Current implementation status:

- **Activation functions** — polymorphic `IActivation` interface with `Sigmoid`, `Tanh`,
  and `ReLU` implementations (`activate(z)` + `derivative(z)`, derivatives evaluated at
  pre-activation `z` so backpropagation can reuse them later)
- **Neuron system** — weight storage, bias handling, and forward calculation
  (`z = w·x + b`, `output = activation(z)`)
- **Layer and network** — combining neurons into layers, chaining layers into a network
  architecture with dimension validation and forward prediction
- **Training system** — loss-function interface with `MSELoss`, optimizer interface with
  `SGD` scaffolding; backpropagation and weight-update passes are on the roadmap
- **Dataset and evaluation** — `Dataset` container with CSV loading and accuracy-metric
  interfaces defined; full ingestion and evaluation passes are on the roadmap

## Architecture

| Component | Responsibility |
|---|---|
| `Neuron` | Smallest computational unit. Stores weights and bias, produces an output. |
| `Layer` | Collection of neurons. Manages neurons and passes outputs forward. |
| `Neural Network` | Combination of layers. Defines architecture and generates predictions. |
| `IActivation` family | Non-linearity strategies, interchangeable through polymorphism. |
| Training module | Loss calculation, backpropagation, and optimization. |
| Dataset / Evaluation | Data handling, metrics, and testing. |

Module ownership:

| Member | Area | Tasks |
|---|---|---|
| 1 | Neural Network Core | Layer and network architecture. |
| 2 | Neuron and Activation System | Neuron implementation and activation functions. |
| 3 | Training System | Loss functions, backpropagation, and optimization. |
| 4 | Dataset, Evaluation and Integration | Data handling, metrics, testing, integration, documentation. |

See `docs/early_prototype_report.tex` (and the compiled PDF next to it) for the full
design write-up with architecture diagrams.

## Project structure

```text
MiniANN_Early/
  CMakeLists.txt
  build.bat
  include/miniann/
    types.hpp        # Vector / Matrix aliases
    activation.hpp   # IActivation, Sigmoid, Tanh, ReLU
    neuron.hpp       # Neuron: weights, bias, forward
    layer.hpp        # Layer: neuron collection, forward
    network.hpp      # NeuralNetwork: architecture + predict
    training.hpp     # ILoss / MSELoss, IOptimizer / SGD interfaces
    dataset.hpp      # Dataset container, loadCSV / accuracy interfaces
  src/               # one .cpp per header
  demos/
    activation_demo.cpp  # activation behavior through base-class pointers
    forward_demo.cpp     # untrained forward pass over XOR inputs
  tests/
    test_activation.cpp  # activation value / derivative assertions
  docs/
    early_prototype_report.tex / .pdf
```

## Getting started

Requirements: Windows + PowerShell, `g++` with C++17 support. CMake is optional.

```bat
cd MiniANN_Early
.\build.bat
```

This builds `activation_demo.exe`, `forward_demo.exe`, and `test_activation.exe`.
CMake alternative:

```bat
cmake -B build -S .
cmake --build build --config Release
```

## Usage

Polymorphic activations:

```cpp
#include "miniann/activation.hpp"
using namespace miniann;

const IActivation* a = new Sigmoid();
a->activate(0.0);    // 0.5
a->derivative(0.0);  // 0.25
```

Building and running a network:

```cpp
#include "miniann/network.hpp"
using namespace miniann;

NeuralNetwork net(42);                       // seeded RNG for init
net.addLayer(2, 2, useTanh());               // hidden layer
net.addLayer(1, 2, useSigmoid());            // output layer

Vector out = net.predict({1.0, 0.0});
```

Loss values:

```cpp
#include "miniann/training.hpp"
MSELoss loss;
double v = loss.compute(out, {1.0});
```

Running the demos:

```bat
.\test_activation.exe
.\activation_demo.exe
.\forward_demo.exe
```

Expected output (abbreviated):

```text
MEETING1 ACTIVATION TESTS PASS
MiniANN Meeting 1: activation module
  z=0.0 -> 0.5000 (deriv 0.2500)
MiniANN Meeting 1: forward-only XOR (UNTRAINED)
  [0,0] -> 0.4371 (mse-vs-0 0.1910)
  [1,1] -> 0.4317 (mse-vs-0 0.1864)
```

## Testing

`tests/test_activation.cpp` asserts known activation values and derivatives
(`sigmoid(0) = 0.5`, `sigmoid'(0) = 0.25`, `tanh(0) = 0`, `relu(-2) = 0`,
`relu'(3) = 1`) including a virtual-dispatch check through `const IActivation*`.

```bat
.\test_activation.exe
```

## Roadmap

1. **Design** — finalize architecture and define interfaces.
2. **Core components** — implement neurons, layers, and forward propagation.
3. **Training** — implement loss functions, backpropagation, and optimizer.
4. **Integration and testing** — validate using datasets and prepare demonstrations.

Planned final capabilities:

- Creating neural networks.
- Training using datasets.
- Evaluating predictions.
- Modular replacement of components (activations, losses, optimizers, metrics).

