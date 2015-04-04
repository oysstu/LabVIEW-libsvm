/// <summary>
///
///	Main entry point for LabVIEW for liblinear
///
/// </summary>

#pragma once

#include <atomic>
#include <linear.h>

#include "LVException.h"
#include "LVTypeDecl.h"

#pragma region TypeDefs
#include <lv_prolog.h>

struct LVlinear_node
{
	int32_t index;
#ifdef _WIN32
	int32_t padding; // Insert padding on 32-bit windows
#endif
	double value;
};

struct LVlinear_problem
{
	int32_t l;
	LVArray_Hdl<float64> y;
	LVArray_Hdl<LVArray_Hdl<LVlinear_node>> x; // Sparse array
	double bias;
};

struct LVlinear_parameter
{
	uint32_t solver_type;
	double eps;
	double C;
	LVArray_Hdl<int32_t> weight_label;
	LVArray_Hdl<double> weight;
	double p;
};

struct LVlinear_model
{
	LVlinear_parameter param;
	int32_t nr_class;
	int32_t nr_feature;
	LVArray_Hdl<double> w;
	LVArray_Hdl<int32_t> label;
	double bias;
};

#include <lv_epilog.h>
#pragma endregion TypeDefs

// Compile-time size checks

// Check that padding matches that of LVsvm_node (TODO: Check x86)
static_assert (sizeof(LVlinear_node) == sizeof(feature_node), "Size of LVSvm_node does not match svm_node.");

// Check that padding is not inserted in between the two-dimensional sparse arrays used
struct _LVlinear_one_element_cluster {
	LVArray_Hdl<LVlinear_node> Cluster;
};
static_assert (sizeof(LVArray_Hdl<LVlinear_node>) == sizeof(_LVlinear_one_element_cluster), "Byte packing in one-element cluster present");

//-- Static variables
static std::atomic<LVUserEventRef *> loggingUsrEv(nullptr);

//
//-- LIBLINEAR API
//

// DLL Export, C API and call convention
#define LVLIBLINEAR_API extern "C" __declspec(dllexport)
#define CALLCONV __cdecl

LVLIBLINEAR_API void	CALLCONV LVlinear_train(lvError *lvErr, const LVlinear_problem *prob_in, const LVlinear_parameter *param_in, LVlinear_model * model_out);

LVLIBLINEAR_API void	CALLCONV LVlinear_cross_validation(lvError *lvErr, const LVlinear_problem *prob_in, const LVlinear_parameter *param_in, const int32_t nr_fold, LVArray_Hdl<double> target_out);

LVLIBLINEAR_API double	CALLCONV LVlinear_predict(lvError *lvErr, const struct LVlinear_model *model_in, const LVArray_Hdl<LVlinear_node> x_in);

LVLIBLINEAR_API double	CALLCONV LVlinear_predict_values(lvError *lvErr, const LVlinear_model  *model_in, const LVArray_Hdl<LVlinear_node> x_in, LVArray_Hdl<double> dec_values_out);

LVLIBLINEAR_API double	CALLCONV LVlinear_predict_probability(lvError *lvErr, const LVlinear_model  *model_in, const LVArray_Hdl<LVlinear_node> x_in, LVArray_Hdl<double> prob_estimates_out);

//-- Print function (used for console output redirection to LabVIEW)
// Logging is global for now
void LVsvm_print_function(const char * message);
LVLIBLINEAR_API void CALLCONV LVlinear_set_logging_userevent(lvError *lvErr, LVUserEventRef *loggingUserEvent_in);
LVLIBLINEAR_API void CALLCONV LVlinear_get_logging_userevent(lvError *lvErr, LVUserEventRef *loggingUserEvent_out);
LVLIBLINEAR_API void CALLCONV LVlinear_delete_logging_userevent(lvError *lvErr, LVUserEventRef *loggingUserEvent_out);

//-- Helper functions

// Assigns the cluster from LabVIEW to a svm_parameter struct
// Arrays are not copied
void LVConvertParameter(const LVlinear_parameter *param_in, parameter *param_out);

// Assigns the LVsvm_model cluster from LabVIEW to svm_model
// Arrays are not copied
void LVConvertModel(const LVlinear_model *model_in, model *model_out);

void LVConvertModel(const model *model_in, LVlinear_model *model_out);
