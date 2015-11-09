### Build instructions
Step-by-step instructions to building the binary components of the library, including dependencies.

First, download the libsvm/liblinear dependencies.
* [libsvm] (https://github.com/cjlin1/libsvm/releases)
* [liblinear](https://github.com/cjlin1/liblinear/releases)
* [libsvm-dense](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#libsvm_for_dense_data)

To avoid changing the include paths, you can unpack these at C:\dev\ or ~\dev\.

## Windows
* Install Visual Studio 2013. Newer versions works if you modify the included scripts and upgrade the project. Both express and community should work, I use community for the native x64 compiler.
* Grab the .bat files under dependencies in this project, and copy each pair to the root folder of libsvm/liblinear/libsvm-dense.
* Run the bat file corresponding to the bitness you're building for. This will build a static library, which we will link against from visual studio. Note that you should not execute both bat files from the same command prompt.
* The dependency paths are declared in property sheets in the project. The LabVIEW path can be set in CommonPaths.props under LabVIEW-common, which defaults to the LabVIEW 2015 standard directory. The path to libsvm/liblinear/libsvm-dense are declared in Paths.props in their respective folders.
* The library should now build and output the DLLs to (Project Root)\LabVIEW\bin\.

## Linux
* Ensure that gcc/g++ 3.8 or later is used, this should be satisfied by the default compiler in most recent distributions. Additionally, you need the development headers/libraries for your distribution, and the cross-build (multilib) libraries if you are cross-compiling.
* Build the library by calling make in the cpp folder (calling make BITNESS=32 builds the x86 library). The dependency/labview paths are declared at the top of the makefile, which can either be modified there or passed to make. There is no intermediate step required to compile the dependencies on Linux.