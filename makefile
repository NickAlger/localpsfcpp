default: all

#### DOES NOT WORK YET. DO NOT USE ####


##################    VVVV    Change these    VVVV    ##################

EIGEN_INCLUDE := /home/nick/anaconda3/envs/fenics3/include/eigen3 # https://eigen.tuxfamily.org/index.php?title=Main_Page
HLIBPRO_DIR := /home/nick/hlibpro-2.9 # https://www.hlibpro.com/

########################################################################

PYFLAGS  = $(shell python3 -m pybind11 --includes)
PYSUFFIX = $(shell python3-config --extension-suffix)

INCLUDE_DIR := ./include
SRC_DIR  := ./src
OBJ_DIR  := ./obj
BUILD_DIR  := ./bin
LIB_DIR  := ./lib
PYTHON_DIR := ./localpsfcpp

CXXFLAGS := -std=c++17 -pthread -lpthread -O3 -Wall
SHAREDFLAGS := -shared -fPIC

ALL_COMPILE_STUFF = $(CXXFLAGS) $(PYFLAGS) -I$(INCLUDE_DIR) -I$(EIGEN_INCLUDE)

BINDINGS_TARGET = localpsfcpp$(PYSUFFIX)

all: $(PYTHON_DIR)/$(BINDINGS_TARGET)
	@echo 'Finished building target: $@'
	@echo ' '

$(PYTHON_DIR)/$(BINDINGS_TARGET): \
 $(SRC_DIR)/pybind11_bindings.cpp \
 $(INCLUDE_DIR)/kdtree.h \
 $(INCLUDE_DIR)/aabbtree.h \
 $(INCLUDE_DIR)/simplexmesh.h \
 $(INCLUDE_DIR)/brent_minimize.h \
 $(INCLUDE_DIR)/ellipsoid.h \
 $(INCLUDE_DIR)/impulse_response.h
	@echo 'Building target: $@'
	g++ -o "$@" "$<" $(CXXFLAGS) $(SHAREDFLAGS) $(PYFLAGS) -I$(INCLUDE_DIR) -I$(EIGEN_INCLUDE)
	@echo 'Finished building target: $@'
	@echo ' '

$(PYTHON_DIR):
	mkdir -p $(PYTHON_DIR)

clean:
	-rm -rf $(PYTHON_DIR)/$(BINDINGS_TARGET)
	-@echo ' '

.PHONY: all clean dependents
