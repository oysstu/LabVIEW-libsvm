![Palette](./Palette.png)

### Description
A LabVIEW wrapper for libsvm (3.20) and liblinear (2.10).
The implementation is thread-safe, which means that multiple cross-validation/training/predicting can be executed simultaneously.

Interfaces to both libsvm sparse and dense is included. The recommendation is to use the dense variant unless you know you need sparseness. This is both due to better performance and a more practical data format as the indices are implicit. Both sparse and dense perform similarly for small number of features. Note however, that performance for the sparse 32-bit library suffers on Windows because the LabVIEW structures are not directly memory compatible with the C++ library, which introduces unnecessary copies. The recommendation is therefore to use a 64-bit LabVIEW installation if the sparse library is needed on Windows.

Sparse is only necessary for datasets consisting of an extremely large number of features, where many features are zeros.

### Installation
1. Download the newest .vip file on the [release page](https://github.com/oysstu/LabVIEW-libsvm/releases)
2. Open and install the file with VI package manager (VIPM)

This includes all necessary binaries and OpenG dependencies (note: internet access required for automatic download of OpenG). The binaries themselves is also supplied in a separate zip archive, for those that prefer to use the library without installing.

The library is developed in LabVIEW 2015, but the distributed VIPM packages are compatible with LabVIEW 2013. On Linux you need to run VIPM as root, and ensure that the target LabVIEW installation allows for TCP/IP connections (Options - Protocols - TCP/IP).

### Dependencies
* libsvm and liblinear binaries (included in the VIPM package or through the zip files)
* OpenG Toolkit VIs (automatically installed by VIPM)
* Windows: [Visual Studio 2013 Visual C++ Redistributable](http://www.microsoft.com/en-us/download/details.aspx?id=40784) (x86 for 32bit LabVIEW, x64 for 64bit LabVIEW)
* Linux: Compiled with GCC/G++-4.8.5 on CentOS7 (libc and libstdc++ must have an compatible ABI)

### Usage
Each sublibrary contains some basic examples. Look at the official libsvm/liblinear documentation if something should be unclear. 
The documentation for the python wrapper in scikit-learn is also very useful.

The library primarily consists of VIs that more or less wrap the functions exposed through the libsvm/liblinear API.
The data structures used for these calls are relatively similar.

### Bugs
Please report any bugs or feature requests either through the issue system or a message.
