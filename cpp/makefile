# Makefile for libsvm/liblinear LabVIEW wrapper
CC ?= gcc
CXX ?= g++
RM ?= rm -f

# Paths
OUT_PATH ?= .
LV_ROOT ?= /usr/local/natinst/LabVIEW-2015-64
LIBSVM_ROOT ?= $(HOME)/dev/libsvm-320
LIBSVM_DENSE_ROOT ?= $(HOME)/dev/libsvm-dense-3.20
LIBLINEAR_ROOT ?= $(HOME)/dev/liblinear-210

## Compilation flags ##
# -c 		Do not run linker after compilation
# -Wall 	Enable common warnings
# -g 		Enable debugging
# -pedantic 	Extra warnings/errors
# -fPIC 	Force position independent code

## Compiler ##
# CFLAGS: For liblinear blas compilation
CFLAGS = -Wall -Wconversion -O3 -fPIC

CPPFLAGS = \
	-I$(LV_ROOT)/cintools \
	-I./LabVIEW-common \
	-c -Wall -fPIC -O3 -pedantic -shared -std=c++14 -Wno-unknown-pragmas

## Linker  ##
LDPATHS = \
	-L$(LV_ROOT)/cintools

LDFLAGS = -Wall -shared -fPIC -m64
LDLIBS = 

COMMON_OBJS=LVUtility.o LVException.o

## Recipes ##

all: LabVIEW-libsvm LabVIEW-libsvm-dense LabVIEW-liblinear

# Common
LVException.o: LabVIEW-common/LVException.cpp LabVIEW-common/LVException.h
	$(CXX) $(CPPFLAGS) LabVIEW-common/LVException.cpp

LVUtility.o: LabVIEW-common/LVUtility.cpp LabVIEW-common/LVUtility.h
	$(CXX) $(CPPFLAGS) LabVIEW-common/LVUtility.cpp

# libsvm

LabVIEW-libsvm: LabVIEW-libsvm.o svm.o $(COMMON_OBJS)
	$(CXX) $(LDFLAGS) $(LDLIBS) LabVIEW-libsvm.o svm.o $(COMMON_OBJS) -o $(OUT_PATH)/LabVIEW-libsvm.so 

LabVIEW-libsvm.o: LabVIEW-libsvm/LabVIEW-libsvm.cpp LabVIEW-libsvm/LabVIEW-libsvm.h
	$(CXX) -I$(LIBSVM_ROOT) $(CPPFLAGS) LabVIEW-libsvm/LabVIEW-libsvm.cpp

svm.o: $(LIBSVM_ROOT)/svm.h $(LIBSVM_ROOT)/svm.cpp
	$(CXX) $(CPPFLAGS) $(LIBSVM_ROOT)/svm.cpp

# libsvm dense
LabVIEW-libsvm-dense: LabVIEW-libsvm-dense.o svm-dense.o $(COMMON_OBJS)
	$(CXX) $(LDFLAGS) $(LDLIBS) LabVIEW-libsvm-dense.o svm-dense.o $(COMMON_OBJS) -o LabVIEW-libsvm-dense.so 

LabVIEW-libsvm-dense.o: LabVIEW-libsvm-dense/LabVIEW-libsvm-dense.cpp LabVIEW-libsvm-dense/LabVIEW-libsvm-dense.h
	$(CXX) -I$(LIBSVM_DENSE_ROOT) $(CPPFLAGS) LabVIEW-libsvm-dense/LabVIEW-libsvm-dense.cpp

svm-dense.o: $(LIBSVM_DENSE_ROOT)/svm.h $(LIBSVM_DENSE_ROOT)/svm.cpp
	$(CXX) $(CPPFLAGS) -D_DENSE_REP $(LIBSVM_DENSE_ROOT)/svm.cpp -o svm-dense.o

# liblinear
LabVIEW-liblinear: LabVIEW-liblinear.o linear.o tron.o blas.a $(COMMON_OBJS)
	$(CXX) $(LDFLAGS) $(LDLIBS) LabVIEW-liblinear.o linear.o tron.o $(COMMON_OBJS) $(LIBLINEAR_ROOT)/blas/blas.a -o LabVIEW-liblinear.so

LabVIEW-liblinear.o: LabVIEW-liblinear/LabVIEW-liblinear.cpp LabVIEW-liblinear/LabVIEW-liblinear.h
	$(CXX) $(CPPFLAGS) -I$(LIBLINEAR_ROOT) LabVIEW-liblinear/LabVIEW-liblinear.cpp

linear.o: $(LIBLINEAR_ROOT)/linear.h $(LIBLINEAR_ROOT)/linear.cpp
	$(CXX) $(CPPFLAGS) $(LIBLINEAR_ROOT)/linear.cpp

tron.o: $(LIBLINEAR_ROOT)/tron.h $(LIBLINEAR_ROOT)/tron.cpp
	$(CXX) -I$(LIBLINEAR_ROOT) $(CPPFLAGS) $(LIBLINEAR_ROOT)/tron.cpp

blas.a: $(LIBLINEAR_ROOT)/blas/*.c $(LIBLINEAR_ROOT)/blas/*.h
	make -C $(LIBLINEAR_ROOT)/blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';


# Ensures that the cleanup will proceed even if there is a file present with the name clean.
.PHONY: clean dist-clean

# Clean obj
clean:
	find . -name "*.o" -exec $(RM) {} +
	make -C $(LIBLINEAR_ROOT)/blas clean
    
# Clean obj + so
dist-clean: clean
	find . -name "*.so" -exec $(RM) {} +


