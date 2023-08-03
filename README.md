# 2024 Mesh-Motion
This repository is a resource for the Mesh Motion test suite working group in the 2024 High-Fidelity CFD Verification Workshop.


# Cloning the repository

Note, the 2022 workshop repository has been added as a submodule. To clone the 2024 repository along with the 2022 repo submodule:
```
git clone --recurse-submodules https://github.com/HighFidelityCFDVerificationWorkshop/2024MeshMotion 
```

If you have already cloned the repository, you can retrieve the 2022 repository submodule using:
```
git submodule update --init --recursive
```

# TODO
- [x] Finalize test suite motions and test cases ([SciTech 2024 paper](https://highfidelitycfdverificationworkshop.github.io/papers/mesh_motion.pdf))
- [x] Provide mesh motion function implementations and their derivatives in C, Python, and Fortran.
- [x] Generate family of cylinder meshes (Per)
- [x] Document appropriate initial conditions for incompressible tests (James)
- [ ] Set up input format + data-processing (Nathan)
- [ ] Preliminary results (All)

# Data submission

## Data organization
Groups should submit time-histories for each case (Config,Motion,Grid-Index,SolutionOrder-Index,Time-Index) in separate files following the file location and name convention:
```
<GroupName>/<Case>-M<MotionNumber>-h<GridIndex>-p<OrderIndex>-t<TimeIndex>.txt
```
In this convention, `h0` should correspond to the coarsest spatial discretization submitted and `t0` should correspond to the coarsest temporal discretization. `p0` corresponds to first-order accurate discretizations, `p1` corresponds to second-order accurate discretizations, etc.

An example data file location and name would be
```
AFRL/Airfoil-M1-h0-p0-t0.txt
```

The following Motion-Index convention should be used:
- `Cylinder-M1` = 2024 Cylinder, Short-Motion
- `Cylinder-M2` = 2024 Cylinder, Long-Motion
- `Airfoil-M1` = 2024 Airfoil, Heaving
- `Airfoil-M2` = 2024 Airfoil, Heaving + Pitching

Groups should also submit a `<motion>.json` in their submission folder that describes the number of elements corresponding to each `h`-index, as well as the number of solution degrees-of-freedom per equation in a spatial element across `p`-indices. For example, a data submission for cylinder results should include a file `Cylinder.json` with example contents:
```
{"h0":100, "h1":200, "h2":400, "p0":1, "p1":8, "p2":27}
```

## Data format
For each contributed Case/Motion/Resolution, we are requesting time-series data for a set of outputs. Time-integrated quantities will be computed in data-processing by the organizers. Time-histories should include the time-history bounds (i.e. data at initial time t=0 and also data at final time t=1,2, or 40 depending on the test case). Time-histories should include the time-value for each time-instance as well as the requested outputs at each time-instance (Outputs are described in Eqns. 14-17 in the test suite document). Each contributed data-file (representative of a particular Case/Motion/Resolution) should be submitted in comma-separated-value format that consists of a single-line header and time-series data on subsequent lines. Optionally, time-integrated quantities may be submitted by participants by adding a data-entry at the end of the file with the time-value set to NaN. Data should be provided with at least 8-digits of precision. If a requested output is not able to be provided the entry should be filled with value NaN.

The data-header should be the following:
```
Time, Y-Force, Work integrand, Mass, Mass error
```

An example of data file contents for a submission that does not provide 'Mass error' and additionally provides time-integrated outputs would be:
```
Time, Y-Force, Work integrand, Mass, Mass error
0.0000000, 1.5438375, 3.4932846, 3.0829579, NaN
0.2000000, 1.5648394, 3.5349762, 3.0830752, NaN
0.4000000, 1.5740924, 3.8028847, 3.0840783, NaN
0.6000000, 1.5638740, 3.4397543, 3.0892051, NaN
0.8000000, 1.5503957, 3.4932846, 3.0913753, NaN
1.0000000, 1.5400933, 3.4932846, 3.0940148, NaN
NaN, 8.2345720, 25.29479238, 28.2984759, NaN
```


# Working group notes

## 5 July, 2023
- Discussion of more complete data-submission conventions. h,p,t resolution indicators. README.json should be included for each submission describing h,p,t levels.

## 7 June, 2023
- Cancelled last minute due to connectivity issues.
- Per noted that second-derivatives of the motion functions are required for some implementations. Maybe our motion implementations need exteneded...
- Per

## 3 May, 2023
- Discussed updates in data-format and submission guidelines.
- Per detailed his process for scripting cylinder grids. Group decided to move forward with committing the scripted grid generator as well as a particular family of cylinder meshes for participants to use.
- Next steps are generating initial results.

## 5 April, 2023

**Notes:**
- MIT group showed preliminary results of cylinder motion.
- James(MIT) discussed appropriate initial conditions for incompressible cylinder case.
  
**Action items:**
- Per generating + providing cylinder meshes in Gmsh format.
- Nathan working towards adding 2022 data and setting up standardized data-input/processing.
- James adding write-up substantiating incompressible initial conditions.

