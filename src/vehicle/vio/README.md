# Hybrid Smoother/Filter State Estimator

This module contains code for estimating the pose of the robot using stereo camera images, IMU, barometric depth, and acoustic range measurements from a beacon.

## Design Overview

When designing the state estimator, we wanted it to be:
- **Real-time**, and able to update the pose at `50+ Hz` while scaling to long trajectories
- **Robust** enough to recover from loss of vision (e.g no objects nearby in the water)

To meet these requirements, we designed a hybrid state estimation system that combines a fast Kalman filter with a slower (but more robust) factor graph smoother. The smoother does inference over a sliding window of sensor measurements (it's actually a fixed-lag smoother to keep runtime constant), and uses allow these measurements to estimate the most likely robot pose. Unfortunately, it's a little slower than we'd like, and can't update the pose of the vehicle at a high loop rate.

To overcome that limitation, we use a Kalman filter. The Kalman filter is extremely fast, and can run at whatever rate the IMU arrives at (e.g `100 Hz`). This gives a continuous, real-time estimate of pose that's useful for low-level controls. Because it does *filtering* rather than *smoothing*, the Kalman filter is only accurate over short periods of time and can diverge.

We combine these two approaches to get the best of both worlds. The fixed-lag smoother solves the factor graph at whatever rate it can (e.g 1 Hz), and the filter provides a real-time pose in between solves.

The slightly tricky part is synchronizing the filter with the smoother, since the smoother is always lagging behind. When the smoother finishes solving up until time `t`, the filter might be milliseconds or seconds ahead at time `t+`. We have to rewind the filter to time `t`, incorporate the pose estimate from the smoother, and then "re-play" a buffer of sensor measurements from `(t, t+]`. To use a `git` analogy, it's like rebasing a branch on `main`.
