## Project-1 Remote GitHub Development and Performance Monitoring

## Experiment 1 -- C++ Compiler Optimizer Tests  

Q1. Which program is fastest? Is it always the fastest?  
Q2. Which program is slowest? Is it always the slowest? 

### Summary of Debug Builds: MIN_BYTES=100 MAX_BYTES=1000 NUM_BLOCKS+100000
**Builds: OPT=-g and OPT=-O2 -g2
| Program    | Avg (-g)| Avg (-O2-g2) | Speedup mult.  | Notes     
|------------|---------|--------------|----------------|-------------------------------------|
| alloca.out | 0.450   | 0.157        | 2.866x faster  | very fast optimized run             |
| malloc.out | 0.415   | 0.168        | 2.470x faster  | Similar speed up time to alloca.out |
| list.out   | 1.594   | 0.176        | 9.057x faster  | Big increase, but still slower      |
| new.out    | 1.614   | 0.178        | 9.067x faster  | Slowest, but another big improvement| 

Answer 1: In an optimized run (OPT=-O2 -g2), alloca.out is the fastest, but it isn't always. The unoptimized run with the same number of byte range and 10 trials puts malloc.out in the lead by a narrow margin. This shows how optimization can produce different results based on different program memory allocation techniques.

Answer 2: In both an optimized and an unoptimized test, new.out is the slowest of the four programs, but list.out is also slow compared to the other two. It should be noted that in the optimized build of both, a substantial improvement in speed was recorded.

## Experiment 2 -- Data per Node Tests

Q1. Was there a trend in program execution time based on the size of data in each Node? If so, what, and why?


### Optimized build Small Data per Node Test  
(OPT=-O2 -g2), MIN=10, MAX=10, NUM_BLOCKS=100,000, TRIALS=10

| Program    | Avg (s) | Min (s) | Max (s) | Notes                         |
|------------|---------|---------|---------|-------------------------------|
| alloca.out | 0.010   | 0.010   | 0.010   | Fastest                       |
| malloc.out | 0.020   | 0.020   | 0.020   | Second                        |
| list.out   | 0.021   | 0.020   | 0.020   | Third                         |
| new.out    | 0.025   | 0.020   | 0.030   | Fourth                        |  

### Optimized build Medium Data per Node Test
(OPT=-O2 -g2), MIN=100, MAX=1000, NUM_BLOCKS=100,000, TRIALS=10

| Program    | Avg (s) | Min (s) | Max (s) | Notes                         |
|------------|---------|---------|---------|-------------------------------|
| alloca.out | 0.163   | 0.160   | 0.190   | Fastest                       |
| malloc.out | 0.169   | 0.160   | 0.180   | Second                        |
| list.out   | 0.177   | 0.170   | 0.190   | Third                         |
| new.out    | 0.180   | 0.170   | 0.190   | Fourth                        |  

### Optimized build Large Data per Node Test
(OPT=-O2 -g2), MIN=1024, MAX=4096, NUM_BLOCKS=100,000, TRIALS=10

| Program    | Avg (s) | Min (s) | Max (s) | Notes                         |
|------------|---------|---------|---------|-------------------------------|
| alloca.out | 0.788   | 0.760   | 0.840   | Fastest                       |
| malloc.out | 0.792   | 0.730   | 0.820   | Second                        |
| list.out   | 0.809   | 0.630   | 0.910   | Third                         |
| new.out    | 0.821   | 0.750   | 0.840   | Fourth                        | 

Answer. Yes, runtimes increase with larger node sizes. This is because the Nodes contain more bytes to initialize. 

## Experiement 3 -- Block Chain Length Tests

Q1. Was there a trend in program execution time based on the length of the block chain?

### Optimized build Block Chain Length
(OPT=-O2 -g2), MIN=100, MAX=1000, NUM_BLOCKS=10K, TRIALS=10

| Program    | Avg (s) | Min (s) | Max (s) | Notes                         |
|------------|---------|---------|---------|-------------------------------|
| alloca.out | 0.020   | 0.020   | 0.020   | All Same                      |
| malloc.out | 0.020   | 0.020   | 0.020   | All Same                      |
| list.out   | 0.020   | 0.020   | 0.020   | All Same                      |
| new.out    | 0.020   | 0.020   | 0.020   | All Same                      |  

### Optimized build Block Chain Length
(OPT=-O2 -g2), MIN=100, MAX=1000, NUM_BLOCKS=100K, TRIALS=10

| Program    | Avg (s) | Min (s) | Max (s) | Notes                         |
|------------|---------|---------|---------|-------------------------------|
| alloca.out | 0.164   | 0.160   | 0.170   | Tied First                    |
| malloc.out | 0.164   | 0.160   | 0.180   | Tied First                    |
| list.out   | 0.173   | 0.160   | 0.190   | Third                         |
| new.out    | 0.172   | 0.170   | 0.180   | Second                        |  

### Optimized build Block Chain Length
(OPT=-O2 -g2), MIN=100, MAX=1000, NUM_BLOCKS=1M, TRIALS=10

| Program    | Avg (s) | Min (s) | Max (s) | Notes                         |
|------------|---------|---------|---------|-------------------------------|
| alloca.out | 1.614   | 1.590   | 1.700   | Second                        |
| malloc.out | 1.595   | 1.580   | 1.163   | Fastest                       |
| list.out   | 1.672   | 1.510   | 1.740   | Third                         |
| new.out    | 1.673   | 1.630   | 1.720   | Fourth                        | 

Answer 1: 

## Experiment 4 -- Heap Alocations

Q1. Consider heap breaks, what's noticeable?
Q2. Does increasing the stack size affect the heap? Speculate on any similarities and differences in programs?

### Heap breaks (NUM_BLOCKS=100000)
| Program    | brk/sbrk calls  |
|------------|-----------------|
| alloca.out | 69              |
| list.out   | 559             |
| malloc.out | 542             |
| new.out    | 559             |

## Memory Diagram

Q1. Considering either the malloc.cpp or alloca.cpp versions of the program, generate a diagram showing two Nodes. Include in the diagram
the relationship of the head, tail, and Node next pointers.  
- show the size (in bytes) and structure of a Node that allocated six bytes of data
- include the bytes pointer, and indicate using an arrow which byte in the allocated memory it points to.  

## Summary

Q1. There's an overhead to allocating memory, initializing it, and eventually processing (in our case, hashing it). For each program, were any of these tasks the same? Which one(s) were different?  
Q2. As the size of data in a Node increases, does the significance of allocating the node increase or decrease?  





