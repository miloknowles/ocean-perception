echo "*** Generating C# lcmtypes ***"

LCMTYPES_PACKAGE="vehicle"
echo "* Package name: ${LCMTYPES_PACKAGE}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LCMTYPES_BUILD_DIR="${SCRIPT_DIR}/../../../build/vehicle/lcmtypes/${LCMTYPES_PACKAGE}"

PROJECT_DIR="${SCRIPT_DIR}/.."
LCMTYPES_DIR="${PROJECT_DIR}/lcmtypes"

echo "* Going to lcmtypes folder: ${LCMTYPES_DIR}"
cd $LCMTYPES_DIR

echo "* Generating bindings..."
lcm-gen --csharp *.lcm

echo "* Installing to build folder: ${LCMTYPES_BUILD_DIR}"
mv ${LCMTYPES_PACKAGE}/* ${LCMTYPES_BUILD_DIR}
