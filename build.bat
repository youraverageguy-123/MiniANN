@echo off
REM Build MiniANN Meeting-1 snapshot (g++ directly). Run from MiniANN_Early dir.
set INCL=-Iinclude
set SRC=src\activation.cpp src\neuron.cpp src\layer.cpp src\network.cpp src\training.cpp src\dataset.cpp
set FLAGS=-std=c++17 -O2 -Wall -Wextra -Wpedantic
g++ %FLAGS% %INCL% %SRC% demos\activation_demo.cpp -o activation_demo.exe
if errorlevel 1 exit /b 1
g++ %FLAGS% %INCL% %SRC% demos\forward_demo.cpp -o forward_demo.exe
if errorlevel 1 exit /b 1
g++ %FLAGS% %INCL% %SRC% tests\test_activation.cpp -o test_activation.exe
if errorlevel 1 exit /b 1
echo Build OK (Meeting 1: activation + forward only)
