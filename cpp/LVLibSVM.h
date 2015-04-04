/// <summary>
///
///	Main entry point for LabVIEW for libsvm
///
/// </summary>

#pragma once

#include <atomic>
#include <memory>
#include <svm.h>

#include "LVException.h"
#include "LVTypeDecl.h"

/*
TODOs:

1.	Redirect stdout and stderr, used in libsvm, to a file (using freopen("file.txt","w",stdout/stderr)) or user event.

2.	Make the print function user events per-instance

3.	Implement file saving/loading using the libsvm interface. Good idea for portability

4.	Implement user-friendly precomputed kernels (in LabVIEW).

5.	Implement normalization/scaling function in LabVIEW

6.	Create alternative implementation that allows storing an svm_model to be used in svm_predict.
Thus avoiding the repeated data conversion and verification.
This can either be implemented by returning some kind of data handle/pointer back to LabVIEW, which can be used in the next call to predict.

*/

// Redefinitions of libsvm type declarations using LabVIEW handles
#pragma region TypeDefs

// Set byte alignment to match LabVIEW #pragma pack(1)
#include "lv_prolog.h"

// Double is generally 64-bit aligned on 64-bit, and 32-bit aligned on 32-bit
// Win32 is an exception, where double is 64-bit aligned even on 32-bit
struct LVsvm_node
{
	int32_t index;
#ifdef _WIN32
	int32_t padding; // Insert padding on 32-bit windows
#endif
	double value;
};

// Note that there is a single-element struct in between the inner and outer array of x
// This can be ignored, as there is no byte packing (tested later)
struct LVsvm_problem
{
	int32_t l;
	LVArray_Hdl<float64> y;
	LVArray_Hdl<LVArray_Hdl<LVsvm_node>> x; // Sparse array
};

struct LVsvm_parameter {
	uint32_t svm_type;
	uint32_t kernel_type;
	int32_t degree;
	double gamma;
	double coef0;
	double cache_size;
	double eps;
	double C;
	LVArray_Hdl<int32_t> weight_label;
	LVArray_Hdl<double> weight;
	double nu;
	double p;
	LVBoolean shrinking;
	LVBoolean probability;
};

struct LVsvm_model {
	LVsvm_parameter param;
	int32_t nr_class;							// Number of classes
	int32_t l;									// Number of support vectors
	LVArray_Hdl<LVArray_Hdl<LVsvm_node>> SV;	// Support vectors
	LVArray_Hdl<double, 2> sv_coef;				// Support vector coefficients ((nr_classes-1) X SV count)
	LVArray_Hdl<double> rho;					// Bias term
	LVArray_Hdl<double> probA;
	LVArray_Hdl<double> probB;
	LVArray_Hdl<int32_t> sv_indices;
	LVArray_Hdl<int32_t> label;
	LVArray_Hdl<int32_t> nSV;
};

#include "lv_epilog.h"

// Compile-time size checks

// Check that padding matches that of LVsvm_node (TODO: Check x86)
static_assert (sizeof(LVsvm_node) == sizeof(svm_node), "Size of LVSvm_node does not match svm_node.");

// Check that padding is not inserted in between the two-dimensional sparse arrays used
struct _LVsvm_one_element_cluster {
	LVArray_Hdl<LVsvm_node> Cluster;
};
static_assert (sizeof(LVArray_Hdl<LVsvm_node>) == sizeof(_LVsvm_one_element_cluster), "Byte packing in one-element cluster present");

#pragma endregion

//-- Static variables

// User event reference used to return libsvm console logging to LabVIEW
// Atomic because the library is set to be multithreaded to avoid hogging the UI thread for calculations
// int32 is not atomic in itself across all compilers and architectures
static std::atomic<LVUserEventRef *> loggingUsrEv(nullptr);

//
//-- LIBSVM API
//

// DLL Export, C API and call convention
#define LVLIBSVM_API extern "C" __declspec(dllexport)
#define CALLCONV __cdecl

LVLIBSVM_API int32_t	CALLCONV GetLibSVMVersion() { return LIBSVM_VERSION; }

LVLIBSVM_API void		CALLCONV LVsvm_train(lvError *lvErr, const LVsvm_problem *prob_in, const LVsvm_parameter *param_in, LVsvm_model * model_out);

LVLIBSVM_API void		CALLCONV LVsvm_cross_validation(lvError *lvErr, const LVsvm_problem *prob_in, const LVsvm_parameter *param_in, int32_t nr_fold, LVArray_Hdl<double> target_out);

LVLIBSVM_API double		CALLCONV LVsvm_predict(lvError *lvErr, const struct LVsvm_model *model_in, const LVArray_Hdl<LVsvm_node> x_in);

LVLIBSVM_API double		CALLCONV LVsvm_predict_values(lvError *lvErr, const LVsvm_model *model_in, const LVArray_Hdl<LVsvm_node> x_in, LVArray_Hdl<double> dec_values_out);

LVLIBSVM_API double		CALLCONV LVsvm_predict_probability(lvError *lvErr, const LVsvm_model *model_in, const LVArray_Hdl<LVsvm_node> x_in, LVArray_Hdl<double> prob_estimates_out);

//-- File operations (TODO)
// File saving/loading should be done through LabVIEW API, these are included for interoperability
//LVLIBSVM_API int		CALLCONV LVsvm_save_model(lvError *lvErr, const char *model_file_name_in, const svm_model *model_in);
//LVLIBSVM_API void		CALLCONV LVsvm_load_model(lvError *lvErr, const char *model_file_name_in, svm_model *model_out);

//-- Print function (used for console output redirection to LabVIEW)
// Logging is global for now
void LVsvm_print_function(const char * message);
LVLIBSVM_API void CALLCONV LVsvm_set_logging_userevent(lvError *lvErr, LVUserEventRef *loggingUserEvent_in);
LVLIBSVM_API void CALLCONV LVsvm_get_logging_userevent(lvError *lvErr, LVUserEventRef *loggingUserEvent_out);
LVLIBSVM_API void CALLCONV LVsvm_delete_logging_userevent(lvError *lvErr, LVUserEventRef *loggingUserEvent_out);

//-- Helper functions

// Assigns the cluster from LabVIEW to a svm_parameter struct
void LVConvertParameter(const LVsvm_parameter *param_in, svm_parameter *param_out);

void LVConvertParameter(const svm_parameter *param_in, LVsvm_parameter *param_out);

// Assigns the LVsvm_model cluster from LabVIEW to svm_model
void LVConvertModel(const LVsvm_model *model_in, svm_model *model_out, std::unique_ptr<svm_node*[]> &SV, std::unique_ptr<double*[]> &sv_coef);

void LVConvertModel(const svm_model *model_in, LVsvm_model *model_out);
