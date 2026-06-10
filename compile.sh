#!/bin/bash
set -e

if [ -z "$CXX" ]; then
    for candidate in g++-15 g++-14 g++-13 g++-12 g++; do
        if command -v "$candidate" >/dev/null 2>&1; then
            CXX="$candidate"
            break
        fi
    done
fi

CXXFLAGS="-std=c++17 -O2"

if echo '#include <omp.h>
int main() { return 0; }' | "$CXX" -std=c++17 -fopenmp -x c++ - -o /tmp/openmp_test >/dev/null 2>&1; then
    CXXFLAGS="$CXXFLAGS -fopenmp"
    echo "Compiling with OpenMP using $CXX"
else
    echo "OpenMP is not available with $CXX; compiling serial fallback."
fi

"$CXX" $CXXFLAGS src/*.cpp -o poisson
