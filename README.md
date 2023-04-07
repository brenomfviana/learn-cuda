# CUDA code for learning

This repository contains simple CUDA codes created for mere learning.

## TODO

- [x] Addition of vectors
- [x] Matrix multiplication
- [x] Matrix multiplication with shared memory
- [x] Linear interpolation
- [x] Bilinear interpolation

## How to test the codes

### Requirements

- CUDA Toolkit 8.0

### How to compile

```bash
nvcc -I include/ file-name.cu -o program-name.x
```

### How to run

```bash
./program-name.x ARGS
```

## References

[CUDA Toolkit Documentation v8.0](http://docs.nvidia.com/cuda/index.html)
