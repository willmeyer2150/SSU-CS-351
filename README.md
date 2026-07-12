# CS 351 — Computer Architecture

This repository contains my coursework for CS 351 at Sonoma State University. The projects cover several parts of computer architecture, starting with memory allocation and CPU performance before moving into multithreading, graphics shaders, and CUDA programming.

What I liked about this course was seeing how much the hardware matters to the way a program performs. Two programs can produce the same result but behave very differently depending on how they allocate memory, divide work between cores, or move data between the CPU and GPU. Most of these projects use timing experiments and comparisons to make those differences visible.

## Projects

### [Project 1 — Memory Allocation and Performance](project-01-memory-allocation-performance/README.md)

Project 1 compares several ways of building and processing a linked list in C++. The implementations use `alloca`, `malloc`, `new`, and `std::list` so I could test how stack allocation, heap allocation, compiler optimization, payload size, and list length affect runtime.

This project gave me a better picture of the costs behind memory allocation. The programs perform the same general work, but their memory layouts and allocation strategies lead to different performance as the workload grows.

### [Project 2 — Threading and Multicore Applications](project-02-threading-and-multicore/README.md)

Project 2 focuses on splitting computational work across multiple CPU threads. The first program computes the mean of a large data set, while the second uses Monte Carlo sampling and a signed-distance function to estimate the volume of a cube with a sphere removed.

The main lesson here was that more threads do not automatically mean the same amount of additional performance. The mean calculation runs into memory-bandwidth limits fairly early, while the more computationally expensive Monte Carlo workload continues scaling across more cores. This project made concepts like speedup, diminishing returns, and Amdahl's Law feel much more concrete.

### [Project 3 — WebGL and Shader Graphics](project-03-webgl-shader-graphics/README.md)

Project 3 uses WebGL vertex and fragment shaders to draw and animate shapes in the browser. It starts with basic geometry such as triangles and polygons, then moves into stars, animation, and color effects.

This ended up being one of my favorite parts of the course. It was satisfying to see the relationship between the math, vertex data, shaders, and the final image on screen. The smaller examples also gave me room to experiment and understand what each part of the graphics pipeline was doing.

### [Project 6 — CUDA Applications](project-06-cuda-applications/README.md)

Project 6 compares CPU and GPU implementations of two workloads. The first recreates `std::iota` with a CUDA kernel, and the second generates Julia and Mandelbrot fractals by assigning image pixels to GPU threads.

CUDA was the most challenging part of the course for me. I understood the general idea of dividing independent work across GPU threads, but translating an existing CPU loop into CUDA blocks, threads, and coordinates took more practice. The contrast between the two programs helped show why GPUs are a much better fit for some problems than others: a simple memory-bound operation gets limited benefits, while an independent per-pixel calculation has much more useful parallel work.

## Main Topics

- Memory layout, allocation, and cache behavior
- Compiler optimization and performance measurement
- Multicore programming with C++ threads
- Memory bandwidth and parallel scaling
- WebGL vertex and fragment shaders
- CUDA kernels and CPU/GPU data movement
- Workload characteristics and hardware tradeoffs

Each project directory contains its source code, build files, results, and a more detailed README with the experiment notes and reflections.
