export BM_FOLDER='~/blue-meadow'

# Sets up the development environment for Blue Meadow.
alias source_ros='source /opt/ros/melodic/setup.bash'
alias source_bm='source ${BM_FOLDER}/catkin_ws/devel/setup.bash'

# Terminal navigation.
alias bm='cd ${BM_FOLDER}'
alias ws='cd ${BM_FOLDER}/catkin_ws/src/auv/'
alias sim='cd ${BM_FOLDER}/farmsim/'

# Starts up the Unity launcher.
alias unity='/usr/local/bin/UnityHub.AppImage'

source_ros
source_bm
bm
