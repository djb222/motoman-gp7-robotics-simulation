# Industrial Robotics GP7 and Safety Simulation

## Overview

This repository presents my contribution to a university industrial robotics simulation project. The wider project involved a simulated kitchen automation environment with multiple industrial robots, including a UR3, KUKA, ABB, and Yaskawa Motoman GP7.

My work focused on the Motoman GP7 robot and the safety-system logic used in the integrated simulation, including E-stop behaviour, collision checking, safety fencing, and a simulated light curtain.

## My Contribution

This was a group university project. My main contribution focused on the Motoman GP7 robot and the safety-system logic used in the integrated simulation.

I contributed to:

- Creating and integrating the Motoman GP7 robot model into the wider kitchen robotics simulation.
- Defining the GP7 kinematic structure, approximate DH parameters, joint limits, and visual representation.
- Developing GP7 pick-and-stir task logic using inverse kinematics, joint trajectories, and RMRC-style motion.
- Implementing safety-system logic including E-stop gating, reset/resume behaviour, and motion blocking.
- Building a light curtain simulation that detects when robot links enter a protected safety zone.
- Adding collision-checking logic using line-plane intersection and rectangular-prism collider geometry.
- Adding a compact safety fence/barrier around the simulated work area.
- Supporting shared GUI controls for selecting and jogging multiple robots.

## Technologies Used

- Python
- Robotics Toolbox for Python
- SpatialMath
- Swift simulator
- Denavit-Hartenberg modelling
- Forward kinematics
- Inverse kinematics
- Joint-space trajectory planning
- RMRC-style motion
- Collision checking
- Safety-system simulation
- STL mesh assets

## Key Files

```text
src/
├── full_kitchen_simulation.py      # Integrated group simulation with safety systems and GP7 context
├── gp7_environment.py             # GP7 build/integration module
└── motoman_gp7_standalone_demo.py # Standalone GP7 visualisation/demo

meshes/
├── base_gp7.stl
├── shoulder_gp7.stl
├── upperarm_gp7.stl
├── forearm_gp7.stl
├── wrist1_gp7.stl
├── wrist2_gp7.stl
└── wrist3_gp7.stl
