# PolyFlow

An **experimental** MLIR-based compiler and execution engine for workflow languages like Snakemake, etc.

## Motivation

While working on [REANA](https://github.com/reanahub/reana) project at CERN, I noticed that each workflow language has its execution engine: one for local computer, one for Kubernetes, etc. This created problems where we would spend time dealing with different execution engine differences. In my humble opinion, we can do better by having a similar approach that LLVM project took with programming languages - a unified compiler for workflow languages that can translate them to a common representation and then execute a common representation on different execution engines. In this way, we can separate language semantics from the execution engine.

## Building

Please make sure to build LLVM and MLIR projects first according to [the instruction](https://mlir.llvm.org/getting_started/). For this particular project those commands were used to build LLVM and MLIR project (provided as an example):

```sh
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=OFF \
   -DLLVM_CCACHE_BUILD=ON \
   -DLLVM_USE_SANITIZER="Address;Undefined" \
   -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_INCLUDE_TOOLS=ON
cmake --build . --target check-mlir
```

To build PolyFlow, run the following commands:

```sh
mkdir build && cd build
cmake -G Ninja .. -DLLVM_DIR=../llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=../llvm-project/build/lib/cmake/mlir \
  -DLLVM_USE_SANITIZER="Address;Undefined"

cmake --build . --target polyflow-opt
```

*Note:* built LLVM/MLIR project was located in `../llvm-project` directory. Adjust this path according to where you have LLVM/MLIR project.

To run the test, `check-polyflow` target will be usable.

To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```

# Credits

- [MLIR Hello World repo](https://github.com/Lewuathe/mlir-hello)