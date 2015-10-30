#include "LVUtility.h"
#include "LVTypeDecl.h"
#include "LVException.h"
#include "extcode.h"
#include <cstring>

#if defined(_WIN32) || defined(_WIN64)
#include <codecvt>
#endif

void LVWriteStringHandle(LStrHandle &strHandle, const char* c_str, size_t length) {
	// Check if handle state is good.
	if (DSCheckHandle(strHandle) == noErr) {
		// Reserve (resize) enough room for the pascal string.
		MgErr err = DSSetHSzClr(strHandle, (sizeof(int32_t) + length));
		if (err) {
			strHandle = NULL;
			throw LVException(__FILE__, __LINE__, "LabVIEW Memory Manager: Failed to allocate memory for string (Out of memory?)");
		}

		// Update pascal size property, and copy data into LabVIEW memory.
		(*strHandle)->cnt = static_cast<int32>(length);
		MoveBlock(c_str, (*strHandle)->str, length);
	}
	else
	{
		throw LVException(__FILE__, __LINE__, "The string handle passed to LVWriteStringHandle is not valid (out of zone/deleted?)");
	}
}

void LVWriteStringHandle(LStrHandle &strHandle, const char* c_str) {
	// Don't want to pass a nullpointer to strlen, behaviour is undefined.
	size_t length;
	if (c_str == NULL) {
		length = 0;
	}
	else {
		length = std::strlen(c_str);
	}
	LVWriteStringHandle(strHandle, c_str, length);
}

void LVWriteStringHandle(LStrHandle &strHandle, std::string str) {
	LVWriteStringHandle(strHandle, str.c_str(), str.length());
}



void LVWriteStringHandle(LStrHandle &strHandle, std::wstring wstr){	
#if defined(_WIN32) || defined(_WIN64)
	// Setup converter
	std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;

	// Use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
	std::string converted_str = converter.to_bytes(wstr);

	LVWriteStringHandle(strHandle, converted_str.c_str(), converted_str.length());
#else
	// gcc4.x does not yet support the above utilities, coming in gcc5
	// Workarounds exist, but might as well wait for the c++11 implementation
	throw LVException(__FILE__, __LINE__, "wstring conversion not implemented for g++");
#endif
}


