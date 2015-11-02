# Makefile for libsvm/liblinear LabVIEW wrapper
# The makefile is a bit verbose to ensure that compilation of dense and sparse is separated properly
CC ?= gcc
CXX ?= g++
RM ?= rm -rf
BITNESS?=64

# Paths
OUT_PATH ?= ../LabVIEW/bin
OBJ_PATH ?= ./obj
LV64_ROOT ?= /usr/local/natinst/LabVIEW-2015-64
LV32_ROOT ?= /usr/local/natinst/LabVIEW-2015
LIBSVM_ROOT ?= $(HOME)/dev/libsvm-320
LIBSVM_DENSE_ROOT ?= $(HOME)/dev/libsvm-dense-3.20
LIBLINEAR_ROOT ?= $(HOME)/dev/liblinear-210

ifeq ($(BITNESS),64)
BITNESS_FLAG = -m64
LV_ROOT := $(LV64_ROOT)
OUT_PATH := $(OUT_PATH)/x64
OBJ_PATH := $(OBJ_PATH)/x64
else
ifeq ($(BITNESS),32)
BITNESS_FLAG = -m32
LV_ROOT := $(LV32_ROOT)
OUT_PATH := $(OUT_PATH)/x86
OBJ_PATH := $(OBJ_PATH)/x86
else
$(error BITNESS must either be set to 32 or 64 (default 64))
endif
endif

## Compilation flags ##
# -c 		Do not run linker after compilation
# -Wall 	Enable common warnings
# -g 		Enable debugging
# -pedantic 	Extra warnings/errors
# -fPIC 	Force position independent code

## Compiler ##
# CFLAGS: For liblinear blas compilation
CFLAGS := $(BITNESS_FLAG) -Wall -Wconversion -O3 -fPIC

CPPFLAGS := \
	-I$(LV_ROOT)/cintools \
	-I./LabVIEW-common \
	$(BITNESS_FLAG) -c -Wall -fPIC -O3 -pedantic -shared -std=c++11 -Wno-unknown-pragmas

## Linker  ##
LDPATHS := -L$(LV_ROOT)/cintools

LDFLAGS := $(BITNESS_FLAG) -Wall -shared -fPIC
LDLIBS = 

COMMON_OBJS := $(OBJ_PATH)/LVUtility.o $(OBJ_PATH)/LVException.o

## Targets ##

# Friendly macro reminder
# $< - returns first dependency
# $^ - returns all dependencies
# $@ - returns the current target

all: createoutdir $(OUT_PATH)/LabVIEW-libsvm.so $(OUT_PATH)/LabVIEW-libsvm-dense.so $(OUT_PATH)/LabVIEW-liblinear.so

# Common
$(OBJ_PATH)/LVException.o: LabVIEW-common/LVException.cpp LabVIEW-common/LVException.h
	$(CXX) $(CPPFLAGS) $< -o $@

$(OBJ_PATH)/LVUtility.o: LabVIEW-common/LVUtility.cpp LabVIEW-common/LVUtility.h
	$(CXX) $(CPPFLAGS) $< -o $@

# libsvm
$(OUT_PATH)/LabVIEW-libsvm.so: $(OBJ_PATH)/LabVIEW-libsvm.o $(OBJ_PATH)/svm.o $(COMMON_OBJS)
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $@

$(OBJ_PATH)/LabVIEW-libsvm.o: LabVIEW-libsvm/LabVIEW-libsvm.cpp LabVIEW-libsvm/LabVIEW-libsvm.h
	$(CXX) -I$(LIBSVM_ROOT) $(CPPFLAGS) $< -o $@

$(OBJ_PATH)/svm.o: $(LIBSVM_ROOT)/svm.cpp $(LIBSVM_ROOT)/svm.h
	$(CXX) $(CPPFLAGS) $< -o $@

# libsvm dense
$(OUT_PATH)/LabVIEW-libsvm-dense.so: $(OBJ_PATH)/LabVIEW-libsvm-dense.o $(OBJ_PATH)/svm-dense.o $(COMMON_OBJS)
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $@

$(OBJ_PATH)/LabVIEW-libsvm-dense.o: LabVIEW-libsvm-dense/LabVIEW-libsvm-dense.cpp LabVIEW-libsvm-dense/LabVIEW-libsvm-dense.h
	$(CXX) -I$(LIBSVM_DENSE_ROOT) $(CPPFLAGS) $< -o $@

$(OBJ_PATH)/svm-dense.o: $(LIBSVM_DENSE_ROOT)/svm.cpp $(LIBSVM_DENSE_ROOT)/svm.h
	$(CXX) $(CPPFLAGS) -D_DENSE_REP $< -o $@

# liblinear
$(OUT_PATH)/LabVIEW-liblinear.so: $(OBJ_PATH)/LabVIEW-liblinear.o $(OBJ_PATH)/linear.o $(OBJ_PATH)/tron.o blas.a $(COMMON_OBJS)
	$(CXX) $(LDFLAGS) $(LDLIBS) $(OBJ_PATH)/LabVIEW-liblinear.o $(OBJ_PATH)/linear.o $(OBJ_PATH)/tron.o $(COMMON_OBJS) $(LIBLINEAR_ROOT)/blas/blas.a -o $@

$(OBJ_PATH)/LabVIEW-liblinear.o: LabVIEW-liblinear/LabVIEW-liblinear.cpp LabVIEW-liblinear/LabVIEW-liblinear.h
	$(CXX) $(CPPFLAGS) -I$(LIBLINEAR_ROOT) $< -o $@

$(OBJ_PATH)/linear.o: $(LIBLINEAR_ROOT)/linear.cpp $(LIBLINEAR_ROOT)/linear.h
	$(CXX) $(CPPFLAGS) $< -o $@

$(OBJ_PATH)/tron.o: $(LIBLINEAR_ROOT)/tron.cpp $(LIBLINEAR_ROOT)/tron.h
	$(CXX) -I$(LIBLINEAR_ROOT) $(CPPFLAGS) $< -o $@

# Note: clean is executed every time, as the 32bit and 64bit objects reside in the same folder
blas.a: $(LIBLINEAR_ROOT)/blas/*.c $(LIBLINEAR_ROOT)/blas/*.h
	make -C $(LIBLINEAR_ROOT)/blas clean
	make -C $(LIBLINEAR_ROOT)/blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';


# Ensures that the cleanup will proceed even if a file exists with that name
.PHONY: clean dist-clean createoutdir all

# Default target
.DEFAULT: all

# Clean obj
clean:
	find $(OBJ_PATH) -name "*.o" -exec $(RM) {} +
	make -C $(LIBLINEAR_ROOT)/blas clean
    
# Clean obj + so
dist-clean: clean
	find $(OUT_PATH) -name "*.so" -exec $(RM) {} +

createoutdir:
	mkdir -p "$(OUT_PATH)"
	mkdir -p "$(OBJ_PATH)"

