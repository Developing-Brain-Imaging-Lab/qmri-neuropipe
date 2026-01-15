cmake_minimum_required(VERSION 3.18)
project(qmri_nextgen LANGUAGES CXX)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/external/cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_COMPILER /opt/homebrew/bin/clang)
set(CMAKE_C_COMPILER /opt/homebrew/bin/clang++)

option(QMRI_USE_OPENMP "Enable OpenMP parallelism" ON)


find_package(Ceres REQUIRED)
find_package(ITK REQUIRED)
find_package(NLopt REQUIRED)
find_package(Docopt REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

include(${ITK_USE_FILE})

add_library(qmri_nextgen INTERFACE)
target_include_directories(qmri_nextgen INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/external ${DOCOPT_INCLUDE_DIR} ${OpenMP_CXX_INCLUDE_DIRS}) 
target_link_libraries(qmri_nextgen INTERFACE OpenMP::OpenMP_CXX Eigen3::Eigen Ceres::ceres ${CERES_LIBRARIES} ${ITK_LIBRARIES} ${DOCOPT_LIBRARIES})



add_executable(qmri_fit_despot1 apps/qmri_fit_despot1.cpp)
target_link_libraries(qmri_fit_despot1 qmri_nextgen)

add_executable(qmri_fit_despot1_hifi apps/qmri_fit_despot1_hifi.cpp)
target_link_libraries(qmri_fit_despot1_hifi qmri_nextgen)

add_executable(qmri_fit_despot2 apps/qmri_fit_despot2.cpp)
target_link_libraries(qmri_fit_despot2 qmri_nextgen)

add_executable(qmri_fit_despot2fm apps/qmri_fit_despot2fm.cpp)
target_link_libraries(qmri_fit_despot2fm qmri_nextgen)