![Palette](./Palette.png)

### Description
A LabVIEW wrapper for libsvm (3.22) and liblinear (2.11).
The implementation is thread-safe, which means that multiple cross-validate/train/predict operations can be executed simultaneously.

Interfaces to both libsvm sparse and dense is included. The recommendation is to use the dense variant unless you know you need sparseness. This is both due to better performance and a more practical data format as the indices are implicit. Both sparse and dense perform similarly for small number of features. Note however, that performance for the sparse 32-bit library suffers on Windows because the LabVIEW structures are not directly memory compatible with the C++ library, which introduces unnecessary copies. The recommendation is therefore to use a 64-bit LabVIEW installation if the sparse library is needed on Windows.

Sparse is only necessary for datasets consisting of an extremely large number of features, where many features are zeros.

### Installation
1. Download the newest .vip file on the [release page](https://github.com/oysstu/LabVIEW-libsvm/releases)
2. Open and install the file with VI package manager (VIPM)

This includes all necessary binaries and OpenG dependencies (note: internet access required for automatic download of OpenG). The binaries themselves is also supplied in a separate zip archive, for those that prefer to use the library without installing.

The library is compatible with LabVIEW 2015 SP1 (64/32bit) and later. On Linux you need to run VIPM as root, and ensure that the target LabVIEW installation allows for TCP/IP connections (Options - Protocols - TCP/IP).

### Dependencies
* libsvm and liblinear binaries (included in the VIPM package or through the zip files)
* OpenG Toolkit VIs (automatically installed by VIPM)
* Windows: [Microsoft Visual C++ 2017 Redistributable](https://www.visualstudio.com/downloads/) (x86 for 32bit LabVIEW, x64 for 64bit LabVIEW)
* Linux: Distributed binaries compiled with GCC/G++-5.4 on Ubuntu 16.04 LTS, compatible ABI required on target system

### Usage
Each sub-library contains some basic examples. Look at the official libsvm/liblinear documentation if something should be unclear. 
The documentation for the python wrapper in scikit-learn is also very useful.

The library primarily consists of VIs that more or less wrap the functions exposed through the libsvm/liblinear API.
The data structures used for these calls are relatively similar.

Note that when distributing built executables on Windows with this library, you may have to include msvcp140.dll in your build specification (Windows/System32/msvcp140.dll). If the redistributable is installed on the target system, it should be found automatically.

### Bugs
Please report any bugs or feature requests either through the issue system or a message.
