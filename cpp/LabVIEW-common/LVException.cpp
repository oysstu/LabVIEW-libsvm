#include "LVException.h"

#include <iostream>
#include <string>
#include "extcode.h"

std::ostream &operator<<(std::ostream &stream, LVException &ex) {
	stream << "Exception: " << ex.what() << "\n";
	stream << "In file: " << ex.getFile() << "\n";
	stream << "On line: " << ex.getLine() << "\n";

	return stream;
}

void LVException::populateErrorCluster(lvError * err) {
	std::string debugInfo;
	err->code = m_code;
	err->status = true;

	if (m_debug){
		debugInfo = "<<DEBUGINFO: File=";
		debugInfo += m_file;
		debugInfo += " Line=";
		debugInfo += std::to_string(m_line);
		debugInfo += ">> ";
		m_messageLength += debugInfo.size();
	}

	// Check if the existing string handle is valid.
	if (DSCheckHandle(err->source) == mgNoErr) {
		// Handle valid: Check if it needs to be resized
		if ((m_messageLength + sizeof(int32_t)) > DSGetHandleSize(err->source)) {
			DSSetHandleSize(err->source, sizeof(int32_t) + m_messageLength);
		}
	}
	else {
		// Handle invalid: Create a new string handle
		err->source = (LStrHandle)DSNewHandle(sizeof(int32_t) + m_messageLength);
	}
	if (m_debug) {
		// Move the string data (with debug info)
		debugInfo += this->what();
		MoveBlock(debugInfo.c_str(), (*err->source)->str, m_messageLength);
	}
	else {
		// Move the string data (without debug info)
		MoveBlock(this->what(), (*err->source)->str, m_messageLength);
	}

	// Set the string length variable
	(*err->source)->cnt = static_cast<int32>(m_messageLength);
}

void LVException::postLVErrorEvent(LVUserEventRef * errorUserEvent) {
	lvError errorCluster;
	populateErrorCluster(&errorCluster);

	// Post to the Event structure
	PostLVUserEvent(*errorUserEvent, &errorCluster);

	// Dispose of the string handle.
	if (errorCluster.source)
		DSDisposeHandle(errorCluster.source);
}

void LVException::returnError(lvError * err) {
	populateErrorCluster(err);
}

void LVException::returnStdException(lvError * lvErr, const char * file, const int line, std::exception &ex){
	std::string msg = "Std exception: ";
	msg += ex.what();
	LVException lvex(file, line, msg);
	lvex.returnError(lvErr);
}
