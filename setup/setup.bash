#-------------------------------------------------------------------------------
# Sets up a development environment for Blue Meadow.
# ** Assumes the following directory structure **
# /home/$USER/bluemeadow
#   catkin_ws
#     src
#       vehicle
#   farmsim
#-------------------------------------------------------------------------------
BM_HOME=$HOME/bluemeadow
CATKIN_WS=${BM_HOME}/catkin_ws

alias source_ros="source /opt/ros/melodic/setup.bash"
alias source_bm="source ${BM_HOME}/catkin_ws/devel/setup.bash"

# Terminal navigation.
alias bm="cd ${BM_HOME}"
alias ws="cd ${CATKIN_WS}/src/vehicle/"
alias bin="cd ${CATKIN_WS}/devel/lib/vehicle/"
alias sim="cd ${BM_HOME}/farmsim/"

# Starts up the Unity launcher.
alias unity='/usr/local/bin/UnityHub.AppImage'

# Convenient command for rebuilding C# lcmtypes.
alias lcm-gen-csharp="${CATKIN_WS}/src/vehicle/setup/lcm-gen-cs.sh"

# Tell lcm-spy where to find lcmtypes.jar so that it can decode our custom messages.
CLASSPATH="${CATKIN_WS}/build/vehicle/lcmtypes/vehicle_lcmtypes.jar"
export CLASSPATH

source_ros
source_bm
bm
