![Palette](./Palette.png)

### Description
A LabVIEW wrapper for libsvm (3.20) and liblinear (2.01).
The implementation is thread-safe, which means that multiple cross-validation/training/predicting can be executed simultaneously.

Interfaces to both libsvm sparse and dense is included. The recommendation is to use the dense variant unless you know you need sparseness. This is both due to better performance and a more practical data format as the indices are implicit. Both sparse and dense perform similarly for small number of features. Note however, that performance for the sparse 32-bit library suffers because the LabVIEW structures are not directly memory compatible with the C++ library, which introduces unnecessary copies. The recommendation is therefore to use a 64-bit LabVIEW installation if the sparse library is needed.

Sparse is only necessary for datasets consisting of an extremely large number of features, where many features are zeros.

### Bugs
The current release (0.5) contains some bugs that will be fixed shortly.
* C and gamma is swapped in the grid-search visualization VI, this causes the rates to be mapped to the wrong values (visualization only)
* Average rate grid-search visualization takes the average of the wrong rates.
* The riffle VI present before LabVIEW 2012 is deprecated (due to a biased mean). By supporting LabVIEW 2011, this deprecated VI is used for all newer LabVIEW versions as well. I'm considering restricting support to the last three versions of LabVIEW as a solution to this, and other backwards-incompatibilities.

### Installation
**The easiest way to install the library is through the pre-built VIPM package found under releases.** This installs both the VIs and binary components to user.lib and adds an entry in the LabVIEW palette. Alternatively, one can download the project itself and the binaries under releases corresponding to the target bitness.

The wrapper is developed in LabVIEW 2014, but the distributed VIPM packages are compatible with LabVIEW 2011 and later.

### Dependencies
* libsvm and liblinear binaries (included in the VIPM package or through the zip files)
* OpenG Toolkit VIs (automatically installed by VIPM)
* [Visual Studio 2013 Visual C++ Redistributable](http://www.microsoft.com/en-us/download/details.aspx?id=40784) (x86 for 32bit LabVIEW, x64 for 64bit LabVIEW)

### Usage
Currently there are three examples included in the palette. Look at the official libsvm/liblinear documentation if something should be unclear. 
The documentation for the python wrapper in scikit-learn is also very useful.

The library primarily consists of VIs that more or less wrap the functions exposed through the libsvm/liblinear API.
The data structures used for these calls are relatively similar.

### Bugs
Please report any bugs or feature requests either through the issue system or a message.
