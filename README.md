# Multivariate Normal Sampler

C++ multivariate sampler using Eigen Vectors/Values to translated a N(0, sigma) sampler into an arbitrary M-dimensional sampler

## Dependencies

```
gcc with C++11 standard
pkg-config
gtest (https://github.com/google/googletest)
cmake 3.1 >= 3.1
Eigen >=3.3.7 with Environment Variable EIGEN_ROOT_DIR
```

## Compile

```c
mkdir build
cd build
# NOTE: gcc requirement due to gtest
cmake .. -GNinja -DCMAKE_CXX_COMPILER=g++-7
ninja
```