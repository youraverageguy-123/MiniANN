# MiniANN Early — Mentor Meeting 1: Design Overview (INCOMPLETE)

Initial-phase snapshot of Problem Statement 5. Same MVP shape as `MiniANN_MVP/`,
deliberately **unfinished**: architecture is agreed, only the activation module is done.

## What this PDF claims vs what is implemented

| PDF section | Claim | Status here |
|---|---|---|
| §1–4 Overview, ANN, responsibilities, composition `Network -> Layers -> Neurons -> Activations`, network does not train itself | Design agreed | Headers + docs only |
| §5 Activation module: `IActivation { activate(z), derivative(z) }`, `Sigmoid`, `Tanh`, `ReLU` | **First implemented component** | **DONE** (`include/miniann/activation.hpp`, `src/activation.cpp`) |
| §6 Neuron System (weights, bias, forward) | Planned (Member 2) | Forward-only, uniform init, **no backward/gradients** |
| §6 Layer and Network (combining neurons) | Planned (Member 1) | `forward()`/`predict()` only, **no backward** |
| §6 Training System (loss, backprop, optimizer) | Planned (Member 3) | `MSELoss::compute` only; `SGD::step` **throws `logic_error`** |
| §6 Dataset and Evaluation | Planned (Member 4) | `add()` only; `loadCSV`/`accuracy` **throw** |
| §7 Team of 4 (vs 5 in final MVP) | 4-member division | Mirrored in header ownership comments |
| §8 Roadmap 1.Design 2.Core 3.Training 4.Integration | Step 1 done | This repo = end of step 1 + start of step 2 |
| §9 Final deliverable (create/train/evaluate/replace) | Expected later | **Not present** — see `MiniANN_MVP/` for the finished version |
| §10 Mentor feedback questions | Asked | Left open on purpose |

Deliberately missing vs the full MVP: `backward()`/`zeroGradients()`, gradient
accumulation, Xavier/He, Adam, `Trainer::fit`, mini-batches, CSV ingestion,
`Accuracy`/`ConfusionMatrix`, `ModelSerializer`, `ILogger`, XOR learning.

## Layout

```text
MiniANN_Early/
  CMakeLists.txt
  build.bat
  include/miniann/
    types.hpp activation.hpp neuron.hpp layer.hpp network.hpp
    training.hpp   # ILoss/MSELoss-compute + IOptimizer/SGD stub
    dataset.hpp    # Dataset add() + loadCSV/accuracy stubs
  src/activation.cpp neuron.cpp layer.cpp network.cpp training.cpp dataset.cpp
  demos/activation_demo.cpp forward_demo.cpp
  tests/test_activation.cpp
```

## Build + run (proves incomplete)

```bat
cd MiniANN_Early
.\build.bat
.\test_activation.exe
.\activation_demo.exe
.\forward_demo.exe
```

Expected:

```text
Build OK (Meeting 1: activation + forward only)
MEETING1 ACTIVATION TESTS PASS
MiniANN Meeting 1: activation module
  z=0.0 -> 0.5000 (deriv 0.2500)
  ...
STATUS: activation DONE; training/dataset/metrics are TODO stubs
MiniANN Meeting 1: forward-only XOR (UNTRAINED)
  [0,0] -> 0.4371 ...
  [1,1] -> 0.4317 ...
training stub correctly throws: Meeting 1: SGD::step not implemented (Training phase TODO, Member 3)
STATUS: forward works; backward/optimizer/trainer NOT implemented
```

Note the XOR outputs are effectively random (~0.43 everywhere): forward plumbing
works, but without backprop the network cannot learn. That is the intended
Meeting-1 behavior.

## How to use (early API)

```cpp
#include "miniann/activation.hpp"
#include "miniann/network.hpp"
const IActivation* a = new Sigmoid(); // polymorphism target
a->activate(0.0);   // 0.5
a->derivative(0.0); // 0.25

miniann::NeuralNetwork net(42);
net.addLayer(2, 2, miniann::useTanh());
net.addLayer(1, 2, miniann::useSigmoid());
net.predict({1.0, 0.0}); // forward only, no training API exists yet
```

## Pushing to GitHub

`.gitignore` excludes `*.exe`, `*.o`, `build/`, CMake cache, editor junk.
`git add` only `include src demos tests CMakeLists.txt build.bat README.md docs`.
