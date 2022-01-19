# Script for generating C# bindings of lcmtypes and copying into the Unity simulation.
# Need to run this whenever lcmtypes are changed.

echo "||=============== Generating C# lcmtypes ===============||"

if [[ -z $BM_DID_SOURCE_SETUP ]]; then
  echo " * WARNING: setup.bash has not been sourced!"
  echo " * You should probably run 'bm-shell' to get the right environment variables."
fi

echo " * Package name: ${BM_LCMTYPES_PACKAGE}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LCMTYPES_DIR="${BM_VEHICLE_DIR}/lcmtypes"
LCMTYPES_TMP_BUILD_DIR="${BM_VEHICLE_DIR}/lcmtypes/${BM_LCMTYPES_PACKAGE}"
LCMTYPES_FINAL_BUILD_DIR="${BM_CATKIN_WS}/build/vehicle/lcmtypes"

echo " * Going to lcmtypes folder:"
echo "     ${LCMTYPES_DIR}"
cd $LCMTYPES_DIR

echo " * Generating C# bindings"
lcm-gen --csharp *.lcm

echo " * Installing to catkin build folder:"
echo "     ${LCMTYPES_FINAL_BUILD_DIR}"
cp ${LCMTYPES_TMP_BUILD_DIR}/* ${LCMTYPES_FINAL_BUILD_DIR}

if [[ -z $BM_FARMSIM_DIR ]]; then
  echo " * WARNING: Environment variable BM_FARMSIM_DIR not set, will not install LCM types into FarmSim"
else
  FARMSIM_LCM_DIR="${BM_FARMSIM_DIR}/unity/Assets/3rdParty/lcm-dotnet/lcmtypes"
  echo " * Installing to FarmSim:"
  echo "     ${FARMSIM_LCM_DIR}"
  cp $LCMTYPES_TMP_BUILD_DIR/* $FARMSIM_LCM_DIR
fi

echo " * Cleaning up temporary build files"
rm -rf ${LCMTYPES_TMP_BUILD_DIR}

echo " * Done"
