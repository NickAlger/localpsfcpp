default: all

##################    VVVV    Change these    VVVV    ##################

EIGEN_INCLUDE := /home/nick/anaconda3/envs/fenics3/include/eigen3# https://eigen.tuxfamily.org/index.php?title=Main_Page
HLIBPRO_DIR := /home/nick/hlibpro-2.9# https://www.hlibpro.com/

########################################################################

HLIBPRO_LIB := $(HLIBPRO_DIR)/lib
HLIBPRO_INCLUDE := $(HLIBPRO_DIR)/include
HLIBPRO_FLAGS := $(shell $(HLIBPRO_DIR)/bin/hlib-config --cflags --lflags)

PYFLAGS  = $(shell python3 -m pybind11 --includes)
PYSUFFIX = $(shell python3-config --extension-suffix)

INCLUDE_DIR := ./include
SRC_DIR  := ./src
OBJ_DIR  := ./obj
BUILD_DIR  := ./bin
LIB_DIR  := ./lib
PYTHON_DIR := ./localpsfcpp

LDFLAGS  = -L$(HLIBPRO_LIB)
CXXFLAGS := -std=c++17 -pthread -lpthread -O3 -Wall
SHAREDFLAGS := -shared -fPIC
LIBS := -lhpro -Wl,-rpath,$(HLIBPRO_LIB)

BINDINGS_TARGET = localpsfcpp$(PYSUFFIX)

all: $(PYTHON_DIR)/$(BINDINGS_TARGET) $(BUILD_DIR)/example1 $(BUILD_DIR)/example2
	@echo 'Finished building target: $@'
	@echo ' '

$(PYTHON_DIR)/$(BINDINGS_TARGET): \
 $(SRC_DIR)/pybind11_bindings.cpp \
 $(INCLUDE_DIR)/kdtree.h \
 $(INCLUDE_DIR)/aabbtree.h \
 $(INCLUDE_DIR)/simplexmesh.h \
 $(INCLUDE_DIR)/brent_minimize.h \
 $(INCLUDE_DIR)/ellipsoid.h \
 $(INCLUDE_DIR)/impulse_response.h \
 $(INCLUDE_DIR)/interpolation.h \
 $(INCLUDE_DIR)/product_convolution_kernel.h \
 $(INCLUDE_DIR)/hmatrix.h \
 $(INCLUDE_DIR)/lpsf_utils.h
	@echo 'Building target: $@'
	g++ -o "$@" "$<" $(CXXFLAGS) $(HLIBPRO_FLAGS) $(PYFLAGS) $(SHAREDFLAGS) \
					-I$(INCLUDE_DIR) -I $(HLIBPRO_INCLUDE) -I$(EIGEN_INCLUDE) \
					$(LDFLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

$(BUILD_DIR)/example1: \
 $(SRC_DIR)/example1.cpp \
 $(INCLUDE_DIR)/kdtree.h \
 $(INCLUDE_DIR)/aabbtree.h \
 $(INCLUDE_DIR)/simplexmesh.h \
 $(INCLUDE_DIR)/brent_minimize.h \
 $(INCLUDE_DIR)/ellipsoid.h \
 $(INCLUDE_DIR)/impulse_response.h \
 $(INCLUDE_DIR)/interpolation.h \
 $(INCLUDE_DIR)/product_convolution_kernel.h \
 $(INCLUDE_DIR)/hmatrix.h \
 $(INCLUDE_DIR)/lpsf_utils.h
	@echo 'Building target: $@'
	g++ -o "$@" "$<" $(CXXFLAGS) $(HLIBPRO_FLAGS) $(PYFLAGS) \
					-I$(INCLUDE_DIR) -I $(HLIBPRO_INCLUDE) -I$(EIGEN_INCLUDE) \
					$(LDFLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

$(BUILD_DIR)/example2: \
 $(SRC_DIR)/example2.cpp \
 $(INCLUDE_DIR)/kdtree.h \
 $(INCLUDE_DIR)/aabbtree.h \
 $(INCLUDE_DIR)/simplexmesh.h \
 $(INCLUDE_DIR)/brent_minimize.h \
 $(INCLUDE_DIR)/ellipsoid.h \
 $(INCLUDE_DIR)/impulse_response.h \
 $(INCLUDE_DIR)/interpolation.h \
 $(INCLUDE_DIR)/product_convolution_kernel.h \
 $(INCLUDE_DIR)/hmatrix.h \
 $(INCLUDE_DIR)/lpsf_utils.h
	@echo 'Building target: $@'
	g++ -o "$@" "$<" $(CXXFLAGS) $(HLIBPRO_FLAGS) $(PYFLAGS) \
					-I$(INCLUDE_DIR) -I $(HLIBPRO_INCLUDE) -I$(EIGEN_INCLUDE) \
					$(LDFLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

$(PYTHON_DIR):
	mkdir -p $(PYTHON_DIR)

clean:
	-rm -rf $(PYTHON_DIR)/$(BINDINGS_TARGET)
	-rm -rf $(BUILD_DIR)/example1
	-rm -rf $(BUILD_DIR)/example2
	-@echo ' '

.PHONY: all clean dependents
