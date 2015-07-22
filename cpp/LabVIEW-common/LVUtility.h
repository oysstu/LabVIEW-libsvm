/// <summary>
/// Contains functions used to simplify moving data between LabVIEW and DLL.
/// </summary>

#ifndef LVUTILITY_H_
#define LVUTILITY_H_

#include <string>
#include <extcode.h>

#include "LVTypeDecl.h"

/* Useful extcode functions
DSCheckHandle 		-Checks if a handle is valid.
DSNewHandle			-Creates a new handle.
DSNewHClr			-Creates a new handle initialized to zero.
DSDisposeHandle		-Disposes of a handle (only use on those created in the library).
DSSetHandleSize		-Changes handle size, reallocation might occur.
DSSetHSzClr			-Changes handle size and sets all new space to zero.
NumericArrayResize	-Changes the size of a numeric array. NI recommends this function over the two above.
MoveBlock			-Copies a specified number of bytes to another location. Comparable to the C-function memcpy.
PostLVUserEvent		-Fires a user event in LabVIEW with data attached (asynchronously).
*/

// Translates LabVIEW typecodes from the Long/Word/Quad form to LabVIEW types.
enum LVTypecode
{
	Unknown = 0,
	I8 = NumType::iB,
	I16 = NumType::iW,
	I32 = NumType::iL,
	I64 = NumType::iQ,
	U8 = NumType::uB,
	U16 = NumType::uW,
	U32 = NumType::uL,
	U64 = NumType::uQ,
	SGL = NumType::fS,
	DBL = NumType::fD,
	EXT = NumType::fX,
	CSG = NumType::cS,
	CDB = NumType::cD,
	CXT = NumType::cX
};

// Function to return the correct typecode used in NumericArrayResize
// TODO: Make more general using std:enable_if, std::is_signed/std::is_unsigned, std::is_integral along with size to determine typecode
template<class T>	inline LVTypecode LVGetTypecode()	{ return LVTypecode::Unknown; };
template<> inline LVTypecode LVGetTypecode<int8_t>()	{ return LVTypecode::I8; };
template<> inline LVTypecode LVGetTypecode<int16_t>()	{ return LVTypecode::I16; };
template<> inline LVTypecode LVGetTypecode<int32_t>()	{ return LVTypecode::I32; };
template<> inline LVTypecode LVGetTypecode<int64_t>()	{ return LVTypecode::I64; };
template<> inline LVTypecode LVGetTypecode<uint8_t>()	{ return LVTypecode::U8; };
template<> inline LVTypecode LVGetTypecode<uint16_t>()	{ return LVTypecode::U16; };
template<> inline LVTypecode LVGetTypecode<uint32_t>()	{ return LVTypecode::U32; };
template<> inline LVTypecode LVGetTypecode<uint64_t>()	{ return LVTypecode::U64; };
template<> inline LVTypecode LVGetTypecode<float32>()	{ return LVTypecode::SGL; };
template<> inline LVTypecode LVGetTypecode<float64>()	{ return LVTypecode::DBL; };
template<> inline LVTypecode LVGetTypecode<floatExt>()	{ return LVTypecode::EXT; };
template<> inline LVTypecode LVGetTypecode<cmplx64>()	{ return LVTypecode::CSG; };
template<> inline LVTypecode LVGetTypecode<cmplx128>()	{ return LVTypecode::CDB; };
template<> inline LVTypecode LVGetTypecode<cmplxExt>()	{ return LVTypecode::CXT; };

// Throw in the missing LabVIEW defines for good measure
template<> inline LVTypecode LVGetTypecode<int8>()		{ return LVTypecode::I8; };
//template<> inline LVTypecode LVGetTypecode<int16>()	{ return LVTypecode::I16; };
template<> inline LVTypecode LVGetTypecode<int32>()		{ return LVTypecode::I32; };
//template<> inline LVTypecode LVGetTypecode<int64>()	{ return LVTypecode::I64; };
//template<> inline LVTypecode LVGetTypecode<uInt8>()	{ return LVTypecode::U8;  };
//template<> inline LVTypecode LVGetTypecode<uInt16>()	{ return LVTypecode::U16; };
template<> inline LVTypecode LVGetTypecode<uInt32>()	{ return LVTypecode::U32; };
//template<> inline LVTypecode LVGetTypecode<uInt64>()	{ return LVTypecode::U64; };

/// <summary> Allocates room for the string in the handle and copies data over. </summary>
/// <param name='strHandle'>A valid string handle.</param>
/// <param name='c_str'>The string to copy to LabVIEW.</param>
/// <param name='length'>The length of the string.</param>
void LVWriteStringHandle(LStrHandle &strHandle, const char* c_str, size_t length);

/// <summary> Allocates room for the string in the handle and copies data over. </summary>
/// <param name='strHandle'>A valid string handle.</param>
/// <param name='c_str'>The null-terminated string to copy to LabVIEW.</param>
void LVWriteStringHandle(LStrHandle &strHandle, const char* c_str);

/// <summary> Allocates room for the string in the handle and copies data over. </summary>
/// <param name='strHandle'>A valid string handle.</param>
/// <param name='str'>The string to copy to LabVIEW.</param>
void LVWriteStringHandle(LStrHandle &strHandle, std::string str);

/// <summary>
///		Converts the string to utf8.
///		Then allocates room for the string in the handle and copies data over.
///	</summary>
/// <param name='strHandle'>A valid string handle.</param>
/// <param name='wstr'>The string to copy to LabVIEW.</param>
void LVWriteStringHandle(LStrHandle &strHandle, std::wstring wstr);

/// <summary>
/// Changes the size of an array of numeric.
/// Does not update the dimSize parameter.
///	</summary>
/// <param name='handle'>The array handle.</param>
/// <param name='newSize'>The new number of elements in the array (along all dimensions).</param>
template<class T, int dim>
void LVResizeNumericArrayHandle(LVArray_Hdl<T, dim> &handle, size_t newSize) {
	// Check if handle is valid, allocate new handle if not
	if (DSCheckHandle(handle) != noErr){
		handle = reinterpret_cast<LVArray_Hdl<T, dim>>(DSNewHandle(0));
		if (handle == NULL)
			throw LVException(__FILE__, __LINE__, "Unable to create new handle (out of memory?).");
	}

	// Prototype: MgErr NumericArrayResize (int32 typeCode, int32 numDims, Uhandle *dataHP, int32 totalNewSize).
	LVTypecode type = LVGetTypecode<T>();
	if (type == LVTypecode::Unknown)
		throw LVException(__FILE__, __LINE__, "LVResizeNumericArrayHandle received unsupported type. Use LVResizeComplexArrayHandle instead.");

	MgErr err = NumericArrayResize(type, dim, reinterpret_cast<UHandle*>(&handle), newSize);

	if (err != noErr)
		throw LVException(__FILE__, __LINE__, "Failed to resize numeric array (out of memory?).");
}

/// <summary>
/// Allocates room for an array of complex types (clusters) in LabVIEW.
/// Does not update the dimSize parameter.
/// Use ResizeNumericArray or LVResizeNumericArray for arrays of plain data.
/// The cluster/struct in question should have been declared with the correct alignment.
/// New memory is not initialized to zero.
///	</summary>
/// <param name='handle'>The array handle.</param>
/// <param name='newSize'>The new number of elements in the outer array.</param>
template<class T, int dim>
void LVResizeCompositeArrayHandle(LVArray_Hdl<T, dim> &handle, size_t newSize){
	// Taking the sizeof the entire array cluster automatically accounts for padding on 64-bit LabVIEW
	size_t sizeReq = sizeof(LVArray<T, dim>) + sizeof(T)*(newSize - 1);

	if (handle == nullptr || DSCheckHandle(handle) != noErr){
		handle = reinterpret_cast<LVArray_Hdl<T, dim>>(DSNewHandle(sizeReq));
		if (handle == NULL)
			throw LVException(__FILE__, __LINE__, "Unable to create new handle (out of memory?).");
	}
	else{
		size_t hdlSize = DSGetHandleSize(handle);
		// Don't change handle size if it is already the requested size
		if (hdlSize != sizeReq){
			MgErr err = DSSetHandleSize(handle, sizeReq);
			if (err != noErr)
				throw LVException(__FILE__, __LINE__, "Failed to change handle size (out of memory?)");
		}
	}
}

/// <summary>
/// Allocates room for an array of LabVIEW array handles (restricted to this for convenience).
/// If new size is smaller, overshooting handles are deallocated.
/// Valid handles are not created for new elements.
/// Does not update the dimSize parameter.
/// Use LVResizeNumericArray for arrays of plain data.
/// New memory allocated is initialized to zero to avoid being interpreted as valid handles.
///	</summary>
/// <param name='handle'>The outer array handle.</param>
/// <param name='newSize'>The new number of elements in the outer array.</param>
template<class T, int dim1, int dim2>
void LVResizeHandleArrayHandle(LVArray_Hdl<LVArray_Hdl<T, dim2>, dim1> &handle, size_t newSize){
	// Calculate the size in bytes (accounts for padding)
	size_t sizeReq = sizeof(LVArray<LVArray_Hdl<T, dim2>, dim1>) + sizeof(LVArray_Hdl<T, dim2>)*(newSize - 1);

	// Create new handle if current handle is not valid, else deallocate overshooting handles
	if (handle == nullptr || DSCheckHandle(handle) != noErr){
		// Allocate cleared memory, so that it not interpreted as a valid handle.
		handle = reinterpret_cast<LVArray_Hdl<LVArray_Hdl<T, dim2>, dim1>>(DSNewHClr(sizeReq));
		if (handle == NULL){
			throw LVException(__FILE__, __LINE__, "Failed to allocate new handle (out of memory?)");
		}
	}
	else {
		size_t hdlSize = DSGetHandleSize(handle);
		// Don't change handle size if it is already the requested size
		if (hdlSize < sizeReq){
			// Deallocate valid handles that exceed the new size
			if ((*handle)->dimSize > newSize){
				for (size_t i = (*handle)->dimSize - 1; i > newSize - 1; i--){
					if ((*handle)->elt[i] != nullptr && DSCheckHandle((*handle)->elt[i]) == noErr){
						MgErr err = DSDisposeHandle((*handle)->elt[i]);
						if (err != noErr)
							throw LVException(__FILE__, __LINE__, "Failed to deallocate overshooting handle when resizing array.");
					}
				}
			}

			// Allocate cleared memory, so that it not interpreted as a valid handle.
			MgErr err = DSSetHSzClr(handle, sizeReq);

			if (err != noErr)
				throw LVException(__FILE__, __LINE__, "Failed to change handle size (out of memory?)");
		}
	}
}

#endif // LVUTILITY_H_
