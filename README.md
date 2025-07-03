# Genesis_jetson

This repo contains simplified implementation of laned detecion using nvidia Jetson nano and canny edge detector
with opencv. It extracts flollowing information from detected lane:
- Curvature
- Lateral Deviation
- Relative Yaw Angle

This data is periodicaly sent to main controller (stm) via the uart (uartComms file).

Repo is part of the GenesisProject (included in as a submodule)