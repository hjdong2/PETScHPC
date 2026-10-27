
cmake_minimum_required(VERSION 3.20)

# Use PETSc's MPI wrappers. Must be set BEFORE project().
set(PETSC_DIR  $ENV{PETSC_DIR})
set(PETSC_ARCH $ENV{PETSC_ARCH})
if(EXISTS ${PETSC_DIR}/${PETSC_ARCH}/bin/mpicxx)
  set(CMAKE_C_COMPILER   ${PETSC_DIR}/${PETSC_ARCH}/bin/mpicc)
  set(CMAKE_CXX_COMPILER ${PETSC_DIR}/${PETSC_ARCH}/bin/mpicxx)
else()
  set(CMAKE_C_COMPILER   mpicc)
  set(CMAKE_CXX_COMPILER mpicxx)
endif()

project(poisson_petsc LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:${PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

add_executable(hello src/hello.cpp)
target_link_libraries(hello PRIVATE PkgConfig::PETSC)
target_compile_options(hello PRIVATE -Wall -Wextra -Wpedantic)

add_executable(dmda src/dmda.cpp)
target_link_libraries(dmda PRIVATE PkgConfig::PETSC)
target_compile_options(dmda PRIVATE -Wall -Wextra -Wpedantic)

add_executable(poisson src/poisson.cpp)
target_link_libraries(poisson PRIVATE PkgConfig::PETSC)
target_compile_options(poisson PRIVATE -Wall -Wextra -Wpedantic)
