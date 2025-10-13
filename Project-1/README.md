## Project-1 Remote GitHub Development and Performance Monitoring

### Unoptimized build  
(OPT= -g), MIN=100, MAX=1000, NUM_BLOCKS=100,000, TRIALS=10

| Program    | Avg (s) | Min (s) | Max (s) | Notes                         |
|------------|---------|---------|---------|-------------------------------|
| alloca.out | 0.440   | 0.450   | 0.460   | Second                        |
| malloc.out | 0.380   | 0.415   | 0.430   | Fastest                       |
| list.out   | 1.500   | 1.594   | 1.694   | Third                         |
| new.out    | 1.580   | 1.614   | 1.630   | Fourth                        | 

### Optimized build  
(OPT=-O2 -g2), MIN=100, MAX=1000, NUM_BLOCKS=100,000, TRIALS=10

| Program    | Avg (s) | Min (s) | Max (s) | Notes                         |
|------------|---------|---------|---------|-------------------------------|
| alloca.out | 0.157   | 0.140   | 0.180   | Fastest                       |
| malloc.out | 0.168   | 0.160   | 0.190   | Second                        |
| list.out   | 0.176   | 0.160   | 0.200   | Third                         |
| new.out    | 0.178   | 0.170   | 0.190   | Fourth                        |  

Q. Which program is fastest? Is it always the fastest?  

A. In an optimized run (OPT=-O2 -g2), alloca.out is clearly the fastest, but it isn't always. The unoptimized run with the same number of byte range and 10 trials puts malloc.out in the lead by a narrow margin. This shows how optimization can produce different results based on different program memory allocation techniques.

Q. Which program is slowest? Is it always the slowest? 

A. In both an optimized and an unoptimized test, new.out is the slowest of the four programs, but list.out is also slow compared to the other two.

### Optimized build Node Test  
(OPT=-O2 -g2), MIN=10, MAX=10, NUM_BLOCKS=100,000, TRIALS=10

| Program    | Avg (s) | Min (s) | Max (s) | Notes                         |
|------------|---------|---------|---------|-------------------------------|
| alloca.out | 0.010   | 0.010   | 0.010   | Fastest                       |
| malloc.out | 0.020   | 0.021   | 0.030   | Tied with list.out            |
| list.out   | 0.020   | 0.021   | 0.030   | Tied with malloc.out          |
| new.out    | 0.020   | 0.025   | 0.030   | Third                         |  

### Optimized build - To do, fill in values  
(OPT=-O2 -g2), MIN=100, MAX=1000, NUM_BLOCKS=100,000, TRIALS=10

| Program    | Avg (s) | Min (s) | Max (s) | Notes                         |
|------------|---------|---------|---------|-------------------------------|
| alloca.out | 0.157   | 0.140   | 0.180   | Fastest                       |
| malloc.out | 0.168   | 0.160   | 0.190   | Second                        |
| list.out   | 0.176   | 0.160   | 0.200   | Third                         |
| new.out    | 0.178   | 0.170   | 0.190   | Fourth                        |  

### Optimized build To do, fill in values
(OPT=-O2 -g2), MIN=1024, MAX=4096, NUM_BLOCKS=100,000, TRIALS=10

| Program    | Avg (s) | Min (s) | Max (s) | Notes                         |
|------------|---------|---------|---------|-------------------------------|
| alloca.out | 0.157   | 0.140   | 0.180   | Fastest                       |
| malloc.out | 0.168   | 0.160   | 0.190   | Second                        |
| list.out   | 0.176   | 0.160   | 0.200   | Third                         |
| new.out    | 0.178   | 0.170   | 0.190   | Fourth                        |  

Was there a trend in program execution time based on the size of data in each Node? If so, what, and why?  
Was there a trend in program execution time based on the length of the block chain?  
Consider heap breaks, what's noticeable? Does increasing the stack size affect the heap? Speculate on any similarities and differences in programs?  
Considering either the malloc.cpp or alloca.cpp versions of the program, generate a diagram showing two Nodes. Include in the diagram
the relationship of the head, tail, and Node next pointers.  
show the size (in bytes) and structure of a Node that allocated six bytes of data
include the bytes pointer, and indicate using an arrow which byte in the allocated memory it points to.  
There's an overhead to allocating memory, initializing it, and eventually processing (in our case, hashing it). For each program, were any of these tasks the same? Which one(s) were different?  
As the size of data in a Node increases, does the significance of allocating the node increase or decrease?  




### Heap breaks (NUM_BLOCKS=100,000)

| Program    | brk/sbrk calls |
|------------|-----------------|
| alloca.out | 69              |
| list.out   | 559             |
| malloc.out | 542             |
| new.out    | 559             |
