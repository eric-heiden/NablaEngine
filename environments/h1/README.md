# H1 locomotion model

The model keeps the floating base and all 19 actuated joints (26 position coordinates, 25 velocity DOFs). The minimal collision model has two convex foot meshes and one convex torso mesh against the floor. The CUDA implementation evaluates full articulated mass and bias, implicit actuators, joint limits and frictional contacts. Each of four launches advances two 2.5 ms integrations, refreshing contacts every integration.

Actions are absolute joint targets in this order:

```text
left hip yaw, roll, pitch; left knee; left ankle
right hip yaw, roll, pitch; right knee; right ankle
torso
left shoulder pitch, roll, yaw; left elbow
right shoulder pitch, roll, yaw; right elbow
```

`Environment('h1').default_action` contains the nominal standing targets. The headless runner holds these targets; locomotion needs a policy supplied through `Environment.step(action)`. See the [environment instructions](../README.md) for installation, baseline selection and timing.

The model originates from Newton's Unitree H1 asset. Its inline collision meshes and physical description retain the [Unitree BSD 3-Clause license](../licenses/unitree-BSD-3-Clause.txt).
