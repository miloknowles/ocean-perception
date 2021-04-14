#-------------------------------------------------------------------------------
# Sets up a development environment for Blue Meadow.
# ** Assumes the following directory structure **
# /home/$USER/bluemeadow
#   catkin_ws
#     src
#       vehicle
#   farmsim
#-------------------------------------------------------------------------------
THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export BM_HOME=$HOME/bluemeadow
export BM_CATKIN_WS=${BM_HOME}/catkin_ws
export BM_FARMSIM_DIR=${BM_HOME}/farmsim
export BM_VEHICLE_DIR=${BM_CATKIN_WS}/src/vehicle
export BM_DATASETS_DIR=${HOME}/datasets

alias source_ros="source /opt/ros/melodic/setup.bash"
alias source_bm="source ${BM_CATKIN_WS}/devel/setup.bash"

# Terminal navigation.
alias bm="cd ${BM_HOME}"
alias ws="cd ${BM_VEHICLE_DIR}"
alias bin="cd ${BM_CATKIN_WS}/devel/lib/vehicle/"
alias sim="cd ${BM_FARMSIM_DIR}"

# Add bluemeadow/local to the PATH.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${BM_HOME}/local/lib"
export PATH="${PATH}:${BM_HOME}/local/include:${BM_HOME}/local/bin:${BM_HOME}/local/lib"

# Starts up the Unity launcher.
alias unity='/usr/local/bin/UnityHub.AppImage'

# Convenient command for rebuilding C# lcmtypes.
export BM_LCMTYPES_PACKAGE="vehicle"
alias lcm-gen-cs="${BM_CATKIN_WS}/src/vehicle/setup/lcm-gen-cs.sh"

# Tell lcm-spy where to find lcmtypes.jar so that it can decode our custom messages.
CLASSPATH="${BM_CATKIN_WS}/build/vehicle/lcmtypes/vehicle_lcmtypes.jar"
export CLASSPATH

source_ros
source_bm
bm

export BM_DID_SOURCE_SETUP=1
