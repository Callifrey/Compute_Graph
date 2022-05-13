# C++_Compute_Graph
Simaple compute graph implement based on c++
### Compiling environment
* gcc version 7.3.0 (x86_64-win32-seh-rev0, Built by MinGW-W64 project)

### Complie & run

**1. Vanillia implement**

```bash
g++ .\vanillia_implement.cpp -o vanillia_implement
.\vanillia_implement.exe
```

**output:**

>Result Shape: 32 64 56 56
>Time Consume: 2454 ms

**2. Optimize implement**

```bash
g++ -fopenmp  .\optimize_implement.cpp -o optimize_implement
.\optimize_implement.exe 
```

**output:**

>Result Shape: 32 64 56 56
>Time Consume: 1340 ms

**3. Optimize by g++**

```bash
g++ -fopenmp -O3 .\optimize_implement.cpp -o optimize_implement
.\optimize_implement.exe 
```

**output:**

>Result Shape: 32 64 56 56
>Time Consume: 211 ms
  
