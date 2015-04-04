# LabVIEW-libsvm
A LabVIEW wrapper for libsvm (320). An interface to liblinear (1.19) is also included.
The wrapper can currently be considered experimental, as it has not been extensively tested.

The easiest way to install the library is through the VIPM package.
This installs both the VIs and binary components to vi.lib.
Currently the only supported system is windows, but unix support will be added shortly.

The supplied binaries are compiled using Visual Studio 2013.
The wrapper is developed in LabVIEW 2014, but the distributed VIPM packages are compatible with LabVIEW 2011 and later.
Let me know if you would like me to extend support for previous versions.

Currently there are only two examples. Look at the libsvm/liblinear documentation if something should be unclear. The documentation for the python wrapper in scikit-learn is also very useful