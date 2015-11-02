/// <summary>
///
/// Exception handler for LabVIEW related code.\n
/// Contains functionality to return exceptions as errors in a labVIEW error cluster.
///
/// </summary>

#ifndef LVEXCEPTION_H_
#define LVEXCEPTION_H_

// LabVIEW 32-bit checks for a deprecated define in platdefines.h under linux
#if defined(__linux__) && defined(__i386) && !defined(i386)
	#define i386 1
#endif

#include <stdexcept>
#include <extcode.h>

#include <lv_prolog.h> // Set labVIEW byte padding

//! Representation of a labVIEW error cluster.
struct lvError{
	LVBoolean status; 	//!< True if error, false if warning/nothing.
	int32_t code;		//!< LabVIEW error code.
	LStrHandle source;	//!< Error message.
	lvError() { status = false; code = 0; source = NULL; }
};

typedef lvError **lvErrorHdl;

#include <lv_epilog.h> // Revert to default byte padding

//! Exception class with utility available to forward errors to labVIEW.
class LVException : public std::runtime_error {
public:
	//! Constructor without error code specified (default code: 7000).
	explicit LVException(const char * file, const int line, const std::string msg)
		:
		std::runtime_error(msg),
		m_messageLength(msg.length()),
		m_file(file),
		m_line(line),
		m_code(7000),
#if defined(DEBUG) || defined(_DEBUG)
		m_debug(true)
#else
		m_debug(false)
#endif
	{/* Empty */
	}

	//! Constructor with custom error code
	explicit LVException(const char * file, const int line, const int code, const std::string msg)
		:
		std::runtime_error(msg),
		m_messageLength(msg.length()),
		m_file(file),
		m_line(line),
		m_code(code),
#if defined(DEBUG) || defined(_DEBUG)
		m_debug(true)
#else
		m_debug(false)
#endif
	{/* Empty */
	}

	//! Destructor.
	virtual ~LVException() throw() {};

	//! Static function that returns a std::exception directly
	static void returnStdException(lvError * lvErr, const char * file, const int line, std::exception &ex);

	//! Operator overload (for console output).
	friend std::ostream &operator<<(std::ostream &stream, LVException &ex);

	//! Returns the file where the exception was thrown.
	const char * getFile() const { return m_file.c_str(); }

	//! Returns the line where the exception was thrown.
	const int getLine() const { return m_line; }

	//! Inserts the exception into a labVIEW error cluster using a user event (subscription/asynchronous).
	//! @param errorUserEvent A reference to a user event in labview.
	void postLVErrorEvent(LVUserEventRef * errorUserEvent);

	//! Inserts the exception into a labVIEW error cluster (used for all synchronous calls).
	//! @param err A pointer to a labVIEW error cluster.
	void returnError(lvError * err);

	//! If this flag is set to true, line and file info will be appended to the message regardless of debug/release.
	void addDebugInfo(bool debug) { m_debug = debug; }

private:
	size_t m_messageLength; //!< The length is stored, because std::exception uses C-strings.
	std::string m_file;		//!< The name of the file.
	int m_line;				//!< The line number.
	int m_code;				//!< The labVIEW error code.
	bool m_debug;			//!< If true line and file info will be added to the message.

	void populateErrorCluster(lvError * err);
};

#endif /* LVEXCEPTION_H_ */
