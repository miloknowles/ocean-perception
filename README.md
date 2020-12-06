# AUV

All of the ROS nodes that run on the vehicle go here.

## First-Time Setup

Add these lines to your `.bashrc`:
```bash
alias source_ros='source /opt/ros/melodic/setup.bash'

# Make it easier to launch Unity on Ubuntu.
alias unity='/usr/local/bin/UnityHub.AppImage'

# Run this once before developing in a new terminal window.
alias bminit='source ~/blue-meadow/catkin_ws/src/auv/setup/setup.bash'
```
