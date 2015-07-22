/// <summary>
/// Array handles and string handles are structs of a size parameter (one for each dimension) and the data itself.
/// This file provides templates that avoids having to define all these structs manually.'
/// </summary>

#pragma once

#include <type_traits>

#include "extcode.h"

// Sets byte alignment to match LabVIEW (#pragma pack(1))
// 64-bit LabVIEW should not be affected by this
// Note: the type T must be declared with the correct byte packing elsewhere.
#include "lv_prolog.h"

// LabVIEW Template Array, default dimension 1
template <class T, int dim = 1>
struct LVArray {
	static_assert(std::is_pod<T>::value, "LVArray type must be Plain-Old-Data (POD).");

	uint32_t dimSize[dim]; // Dimension size specifier
	T elt[1];  // The Data array
};

// Specialization for 1D-arrays (non-array dimsize).
template <class T>
struct LVArray < T, 1 > {
	static_assert(std::is_pod<T>::value, "LVArray type must be Plain-Old-Data (POD).");
	uint32_t dimSize;
	T elt[1];
};

// Typedef for LabVIEW array pointers
template <class T, int dim = 1> 
using LVArray_Ptr = LVArray < T, dim >*;

// Typedef for LabVIEW array handles (this is usually used)
template <class T, int dim = 1>
using LVArray_Hdl = LVArray < T, dim >**;

// Return byte padding to default
#include "lv_epilog.h"
