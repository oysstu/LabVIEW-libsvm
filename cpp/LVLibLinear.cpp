#include "stdafx.h"
#include "LVLibLinear.h"

#include <stdint.h>
#include <exception>
#include <string>
#include <memory>
#include <extcode.h>
#include <svm.h>

#include "LVTypeDecl.h"
#include "LVUtility.h"
#include "LVException.h"

void LVlinear_train(lvError *lvErr, const LVlinear_problem *prob_in, const LVlinear_parameter *param_in, LVlinear_model * model_out){
	try{
		// Input verification: Problem dimensions
		if ((*(prob_in->x))->dimSize != (*(prob_in->y))->dimSize)
			throw LVException(__FILE__, __LINE__, "The problem must have an equal number of labels and feature vectors (x and y).");

		//-- Convert problem
		std::unique_ptr<problem> prob(new problem);
		uint32_t nr_nodes = (*(prob_in->y))->dimSize;
		prob->l = nr_nodes;
		prob->y = (*(prob_in->y))->elt;

		// Create and array of pointers (sparse datastructure)
		std::unique_ptr<feature_node*[]> x(new feature_node*[nr_nodes]);
		prob->x = x.get();

		auto x_in = prob_in->x;
		for (unsigned int i = 0; i < (*x_in)->dimSize; i++){
			// Assign the innermost svm_node array pointers to the array of pointers
			auto xi_in_Hdl = (*x_in)->elt[i];
			x[i] = reinterpret_cast<feature_node*>((*xi_in_Hdl)->elt);
		}

		//-- Convert parameters
		std::unique_ptr<parameter> param(new parameter());
		LVConvertParameter(param_in, param.get());

		// Verify parameters
		const char * param_check = check_parameter(prob.get(), param.get());
		if (param_check != nullptr)
			throw LVException(__FILE__, __LINE__, "Parameter check failed with the following error: " + std::string(param_check));

		// Train model
		model *result = train(prob.get(), param.get());

		// Copy model to LabVIEW memory
		LVConvertModel(result, model_out);

		// Release memory allocated by train
		free_model_content(result);
	}
	catch (LVException &ex) {
		ex.returnError(lvErr);
		// To avoid LabVIEW reading and utilizing bad memory, the dimension sizes of arrays is set to zero
		(*(model_out->label))->dimSize = 0;
		(*(model_out->w))->dimSize = 0;
	}
	catch (std::exception &ex) {
		LVException::returnStdException(lvErr, __FILE__, __LINE__, ex);
		(*(model_out->label))->dimSize = 0;
		(*(model_out->w))->dimSize = 0;
	}
	catch (...) {
		LVException ex(__FILE__, __LINE__, "Unknown exception has occurred");
		ex.returnError(lvErr);
		(*(model_out->label))->dimSize = 0;
		(*(model_out->w))->dimSize = 0;
	}
}

void LVlinear_cross_validation(lvError *lvErr, const LVlinear_problem *prob_in, const LVlinear_parameter *param_in, const int32_t nr_fold, LVArray_Hdl<double> target_out){
	try{
		// Input verification: Problem dimensions
		if ((*(prob_in->x))->dimSize != (*(prob_in->y))->dimSize)
			throw LVException(__FILE__, __LINE__, "The problem must have an equal number of labels and feature vectors (x and y).");

		// Convert LVsvm_problem to svm_problem
		std::unique_ptr<problem> prob(new problem);
		uint32_t nr_nodes = (*(prob_in->y))->dimSize;
		prob->l = nr_nodes;
		prob->y = (*(prob_in->y))->elt;

		// Create and array of pointers (sparse datastructure)
		std::unique_ptr<feature_node*[]> x(new feature_node*[nr_nodes]);
		prob->x = x.get();

		auto x_in = prob_in->x;
		for (unsigned int i = 0; i < (*x_in)->dimSize; i++){
			// Assign the innermost svm_node array pointers to the array of pointers
			auto xi_in_Hdl = (*x_in)->elt[i];
			x[i] = reinterpret_cast<feature_node*>((*xi_in_Hdl)->elt);
		}

		// Assign parameters to svm_parameter
		std::unique_ptr<parameter> param(new parameter());
		LVConvertParameter(param_in, param.get());

		// Verify parameters
		const char * param_check = check_parameter(prob.get(), param.get());
		if (param_check != nullptr)
			throw LVException(__FILE__, __LINE__, "Parameter check failed with the following error: " + std::string(param_check));

		// Allocate room in target_out
		LVResizeNumericArrayHandle(target_out, nr_nodes);

		// Run cross validation
		cross_validation(prob.get(), param.get(), nr_fold, (*target_out)->elt);
		(*target_out)->dimSize = nr_nodes;
	}
	catch (LVException &ex) {
		ex.returnError(lvErr);
		(*target_out)->dimSize = 0;
	}
	catch (std::exception &ex) {
		LVException::returnStdException(lvErr, __FILE__, __LINE__, ex);
		(*target_out)->dimSize = 0;
	}
	catch (...) {
		LVException ex(__FILE__, __LINE__, "Unknown exception has occurred");
		ex.returnError(lvErr);
		(*target_out)->dimSize = 0;
	}
}

double LVlinear_predict(lvError *lvErr, const struct LVlinear_model *model_in, const LVArray_Hdl<LVlinear_node> x_in){
	try{
		// Convert LVsvm_model to svm_model
		std::unique_ptr<model> model(new model);
		LVConvertModel(model_in, model.get());

		double label = predict(model.get(), reinterpret_cast<feature_node*>((*x_in)->elt));

		return label;
	}
	catch (LVException &ex) {
		ex.returnError(lvErr);
		return std::nan("");
	}

	catch (std::exception &ex) {
		LVException::returnStdException(lvErr, __FILE__, __LINE__, ex);
		return std::nan("");
	}
	catch (...) {
		LVException ex(__FILE__, __LINE__, "Unknown exception has occurred");
		ex.returnError(lvErr);
		return std::nan("");
	}
}

double LVlinear_predict_values(lvError *lvErr, const LVlinear_model  *model_in, const LVArray_Hdl<LVlinear_node> x_in, LVArray_Hdl<double> dec_values_out){
	try{
		// Convert LVsvm_model to svm_model
		std::unique_ptr<model> model(new model);
		LVConvertModel(model_in, model.get());

		int nr_class = model_in->nr_class;
		int solver = (model_in->param).solver_type;

		// Set up output array
		int nr_dec = 0;
		if (nr_class <= 2){
			if (solver == MCSVM_CS)
				nr_dec = 2;
			else
				nr_dec = 1;
		}
		else {
			nr_dec = nr_class;
		}

		LVResizeNumericArrayHandle(dec_values_out, nr_dec);
		(*dec_values_out)->dimSize = nr_dec;

		double predicted_label = predict_values(model.get(), reinterpret_cast<feature_node*>((*x_in)->elt), (*dec_values_out)->elt);

		return predicted_label;
	}
	catch (LVException &ex) {
		ex.returnError(lvErr);
		(*dec_values_out)->dimSize = 0;
		return std::nan("");
	}

	catch (std::exception &ex) {
		LVException::returnStdException(lvErr, __FILE__, __LINE__, ex);
		(*dec_values_out)->dimSize = 0;
		return std::nan("");
	}
	catch (...) {
		LVException ex(__FILE__, __LINE__, "Unknown exception has occurred");
		ex.returnError(lvErr);
		(*dec_values_out)->dimSize = 0;
		return std::nan("");
	}
}

double LVlinear_predict_probability(lvError *lvErr, const LVlinear_model  *model_in, const LVArray_Hdl<LVlinear_node> x_in, LVArray_Hdl<double> prob_estimates_out){
	try{
		// Convert LVsvm_model to svm_model
		std::unique_ptr<model> model(new model);
		LVConvertModel(model_in, model.get());

		// Check probability model
		int valid_probability = check_probability_model(model.get());
		if (!valid_probability)
			throw LVException(__FILE__, __LINE__, "The model does not support probability output.");

		// Allocate room for probability estimates
		LVResizeNumericArrayHandle(prob_estimates_out, model->nr_class);
		(*prob_estimates_out)->dimSize = model->nr_class;

		double highest_prob_label = predict_probability(model.get(), reinterpret_cast<feature_node*>((*x_in)->elt), (*prob_estimates_out)->elt);

		return highest_prob_label;
	}
	catch (LVException &ex) {
		ex.returnError(lvErr);
		(*prob_estimates_out)->dimSize = 0;
		return std::nan("");
	}

	catch (std::exception &ex) {
		LVException::returnStdException(lvErr, __FILE__, __LINE__, ex);
		(*prob_estimates_out)->dimSize = 0;
		return std::nan("");
	}
	catch (...) {
		LVException ex(__FILE__, __LINE__, "Unknown exception has occurred");
		ex.returnError(lvErr);
		(*prob_estimates_out)->dimSize = 0;
		return std::nan("");
	}
}

//-- Print functions

void LVlinear_print_function(const char * message){
	LVUserEventRef *usrEv = loggingUsrEv;
	if (usrEv != nullptr && message != nullptr){
		// Filter out the progress messages (.....)
		if (strcmp(message, ".") != 0){
			// Move the string to a handle
			size_t length = strlen(message);
			LStrHandle lvmsg = (LStrHandle)DSNewHandle(sizeof(int32_t) + length);
			MoveBlock(message, (*lvmsg)->str, length);
			(*lvmsg)->cnt = static_cast<int32>(length);

			// Post the string to the user event
			PostLVUserEvent(*usrEv, &lvmsg);
		}
	}
}

void LVlinear_set_logging_userevent(lvError *lvErr, LVUserEventRef *loggingUserEvent_in) {
	loggingUsrEv = loggingUserEvent_in;
	set_print_string_function(LVlinear_print_function);
}

void LVlinear_get_logging_userevent(lvError *lvErr, LVUserEventRef *loggingUserEvent_out) {
	loggingUserEvent_out = loggingUsrEv;
}

void LVlinear_delete_logging_userevent(lvError *lvErr, LVUserEventRef *loggingUserEvent_out) {
	loggingUserEvent_out = loggingUsrEv;
	loggingUsrEv = nullptr;
}

//-- Helper functions

void LVConvertParameter(const LVlinear_parameter *param_in, parameter *param_out){
	param_out->solver_type = param_in->solver_type;
	param_out->eps = param_in->eps;
	param_out->C = param_in->C;
	param_out->p = param_in->p;

	if ((*(param_in->weight))->dimSize != (*(param_in->weight_label))->dimSize)
		throw LVException(__FILE__, __LINE__, "Parameter error: Number of elements in weight_label and weight does not match.");

	param_out->nr_weight = (*(param_in->weight))->dimSize;

	// Weight label
	if ((*(param_in->weight_label))->dimSize > 0)
		param_out->weight_label = (*(param_in->weight_label))->elt;
	else
		param_out->weight_label = nullptr;

	// Weight
	if ((*(param_in->weight))->dimSize > 0)
		param_out->weight = (*(param_in->weight))->elt;
	else
		param_out->weight = nullptr;
}

void LVConvertModel(const LVlinear_model *model_in, model *model_out){
	// Assign the parameters
	LVConvertParameter(&model_in->param, &model_out->param);

	// Copy assignments
	model_out->nr_class = model_in->nr_class;
	model_out->nr_feature = model_in->nr_feature;
	model_out->bias = model_in->bias;

	// w
	if ((*(model_in->w))->dimSize > 0)
		model_out->w = (*(model_in->w))->elt;
	else
		model_out->w = nullptr;

	// label
	if ((*(model_in->label))->dimSize > 0)
		model_out->label = (*(model_in->label))->elt;
	else
		model_out->label = nullptr;
}

void LVConvertModel(const model *model_in, LVlinear_model *model_out){
	// Convert svm_model to LVsvm_model
	model_out->nr_class = model_in->nr_class;
	model_out->nr_feature = model_in->nr_feature;
	model_out->bias = model_in->bias;

	int nr_class = model_in->nr_class;
	int nr_feature = model_in->nr_feature;

	// Label
	LVResizeNumericArrayHandle(model_out->label, nr_class);
	MoveBlock(model_in->label, (*(model_out->label))->elt, nr_class * sizeof(int32_t));
	(*model_out->label)->dimSize = model_in->nr_class;

	// w
	int32_t nr_w = nr_feature * nr_class;
	// If bias is present, the feature vector increases by one
	if (model_in->bias >= 0)
		nr_w += nr_class;

	LVResizeNumericArrayHandle(model_out->w, nr_w);
	MoveBlock(model_in->w, (*(model_out->w))->elt, nr_w * sizeof(double));
	(*model_out->w)->dimSize = nr_w;
}
