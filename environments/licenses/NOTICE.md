# Model and source notices

- Unitree Go2 robot descriptions: Copyright (c) 2016–2022 HangZhou YuShu TECHNOLOGY CO., LTD. (Unitree Robotics), [BSD 3-Clause](unitree-BSD-3-Clause.txt). The converted robot description also carries [MPL 2.0](unitree-MPL-2.0.txt) terms; the editable MJCF model is distributed as source.
- Unitree H1 robot description and convex collision meshes: Copyright (c) 2016–2022 Unitree Robotics, [BSD 3-Clause](unitree-BSD-3-Clause.txt), converted from the Newton unitree_h1 asset.
- Franka Panda robot descriptions: [Apache 2.0](franka-Apache-2.0.txt). These scenes use primitive collision approximations, without visual mesh packages.
- MuJoCo and MuJoCo Warp collision arithmetic: Apache 2.0, Copyright DeepMind Technologies Limited and The Newton Developers. Notices remain in the CUDA headers.
- NablaEngine helper arithmetic: the upstream repository's Apache 2.0 license applies.

The task MJCF files preserve the robot's numeric physical description, center the world origin, simplify names, and spell out contact damping. Lift object meshes are geometric boxes, a sphere and capsules. The CUDA solver, Newton adapter and runner additions use the repository's Apache 2.0 license; third-party model terms remain applicable to their respective files.
