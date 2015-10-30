#include "LabVIEW-liblinear.h"

#include <stdint.h>
#include <exception>
#include <string>
#include <cstring>
#include <memory>
#include <cmath>
#include <climits>

#include <extcode.h>
#include <linear.h>

#include <LVTypeDecl.h>
#include <LVUtility.h>
#include <LVException.h>

void LVlinear_train(lvError *lvErr, const LVlinear_problem *prob_in, const LVlinear_parameter *param_in, LVlinear_model * model_out){
	try{
		// Input verification: Nonempty problem
		if (prob_in->x == nullptr || (*(prob_in->x))->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Empty problem passed to liblinear_train.");

		// Input verification: Problem dimensions
		if ((*(prob_in->x))->dimSize != (*(prob_in->y))->dimSize)
			throw LVException(__FILE__, __LINE__, "The problem must have an equal number of labels and feature vectors (x and y).");

		uint32_t nr_nodes = (*(prob_in->y))->dimSize;

		// Input validation: Number of feature vectors too large (exceeds max signed int)
		if(nr_nodes > INT_MAX)
			throw LVException(__FILE__, __LINE__, "Number of feature vectors too large (grater than " + std::to_string(INT_MAX) + ")");

		//-- Convert problem
		auto prob = std::make_unique<problem>();
		prob->l = nr_nodes;
		prob->y = (*(prob_in->y))->elt;
		prob->n = 0; // Calculated later
		prob->bias = prob_in->bias;

		// Create and array of pointers (sparse datastructure)
		auto x = std::make_unique<feature_node*[]>(nr_nodes);
		prob->x = x.get();

		auto x_in = prob_in->x;
		for (unsigned int i = 0; i < (*x_in)->dimSize; i++){
			// Assign the innermost svm_node array pointers to the array of pointers
			auto xi_in_Hdl = (*x_in)->elt[i];
			x[i] = reinterpret_cast<feature_node*>((*xi_in_Hdl)->elt);

			// Input validation: Final index -1?
			if ((*xi_in_Hdl)->elt[(*xi_in_Hdl)->dimSize - 1].index != -1)
				throw LVException(__FILE__, __LINE__, "The index of the last element of each feature vector needs to be -1 (liblinear_train).");

			// Calculate the max index
			// This detail is not exposed in LabVIEW, as setting the wrong value causes a crash
			// Second to last element should contain the max index for that feature vector (as they are in ascending order).
			auto secondToLast = (*xi_in_Hdl)->dimSize - 2; // Ignoring -1 index
			auto largestIndex = (*xi_in_Hdl)->elt[secondToLast].index;
			if (secondToLast >= 0 && largestIndex > prob->n)
				prob->n = largestIndex;
		}

		//-- Convert parameters
		auto param = std::make_unique<parameter>();
		LVConvertParameter(*param_in, *param);

		// Verify parameters
		const char * param_check = check_parameter(prob.get(), param.get());
		if (param_check != nullptr)
			throw LVException(__FILE__, __LINE__, "Parameter check failed with the following error: " + std::string(param_check));

		// Train model
		model *result = train(prob.get(), param.get());

		// Copy model to LabVIEW memory
		LVConvertModel(*result, *model_out);

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
		// Input verification: Nonempty problem
		if (prob_in->x == nullptr || (*(prob_in->x))->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Empty problem passed to liblinear_crossvalidation.");

		// Input verification: Problem dimensions
		if ((*(prob_in->x))->dimSize != (*(prob_in->y))->dimSize)
			throw LVException(__FILE__, __LINE__, "The problem must have an equal number of labels and feature vectors (x and y).");

		uint32_t nr_nodes = (*(prob_in->y))->dimSize;
		// Input validation: Number of feature vectors too large (exceeds max signed int)
		if(nr_nodes > INT_MAX)
			throw LVException(__FILE__, __LINE__, "Number of feature vectors too large (grater than " + std::to_string(INT_MAX) + ")");

		// Convert LVsvm_problem to svm_problem
		auto prob = std::make_unique<problem>();
		prob->l = nr_nodes;
		prob->y = (*(prob_in->y))->elt; 
		prob->n = 0; // Calculated later
		prob->bias = prob_in->bias;

		// Create and array of pointers (sparse datastructure)
		auto x = std::make_unique<feature_node*[]>(nr_nodes);
		prob->x = x.get();

		auto x_in = prob_in->x;
		for (unsigned int i = 0; i < (*x_in)->dimSize; i++){
			// Assign the innermost svm_node array pointers to the array of pointers
			auto xi_in_Hdl = (*x_in)->elt[i];
			x[i] = reinterpret_cast<feature_node*>((*xi_in_Hdl)->elt);

			// Input validation: Final index -1?
			if ((*xi_in_Hdl)->elt[(*xi_in_Hdl)->dimSize - 1].index != -1)
				throw LVException(__FILE__, __LINE__, "The index of the last element of each feature vector needs to be -1 (libsvm_crossvalidation).");

			// Calculate the max index
			// This detail is not exposed in LabVIEW, as setting the wrong value causes a crash
			// Second to last element should contain the max index for that feature vector (as they are in ascending order).
			auto secondToLast = (*xi_in_Hdl)->dimSize - 2; // Ignoring -1 index
			auto largestIndex = (*xi_in_Hdl)->elt[secondToLast].index;
			if (largestIndex > prob->n)
				prob->n = largestIndex;
		}

		// n increases by one if bias is present
		if (prob_in->bias >= 0)
			prob->n++;

		// Assign parameters to svm_parameter
		auto param = std::make_unique<parameter>();
		LVConvertParameter(*param_in, *param);

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
		// Input validation: Uninitialized model
		if (model_in == nullptr || model_in->w == nullptr || (*model_in->w)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Uninitialized model passed to liblinear_predict.");

		// Input validation: Empty feature vector
		if (x_in == nullptr || (*x_in)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Empty feature vector passed to liblinear_predict.");

		// Input validation: Final index -1?
		if ((*x_in)->elt[(*x_in)->dimSize - 1].index != -1)
			throw LVException(__FILE__, __LINE__, "The index of the last element of the feature vector needs to be -1 (liblinear_predict).");

		// Convert LVsvm_model to svm_model
		auto mdl = std::make_unique<model>();
		LVConvertModel(*model_in, *mdl);

		double label = predict(mdl.get(), reinterpret_cast<feature_node*>((*x_in)->elt));

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
		// Input validation: Uninitialized model
		if (model_in == nullptr || model_in->w == nullptr || (*model_in->w)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Uninitialized model passed to liblinear_predict_values.");

		// Input validation: Empty feature vector
		if (x_in == nullptr || (*x_in)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Empty feature vector passed to liblinear_predict_values.");

		// Input validation: Final index -1?
		if ((*x_in)->elt[(*x_in)->dimSize - 1].index != -1)
			throw LVException(__FILE__, __LINE__, "The index of the last element of the feature vector needs to be -1 (liblinear_predict_values).");

		// Convert LVsvm_model to svm_model
		auto mdl = std::make_unique<model>();
		LVConvertModel(*model_in, *mdl);

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

		double predicted_label = predict_values(mdl.get(), reinterpret_cast<feature_node*>((*x_in)->elt), (*dec_values_out)->elt);

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
		// Input validation: Uninitialized model
		if (model_in == nullptr || model_in->w == nullptr || (*model_in->w)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Uninitialized model passed to liblinear_predict_probability.");

		// Input validation: Empty feature vector
		if (x_in == nullptr || (*x_in)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Empty feature vector passed to liblinear_predict_probability.");

		// Input validation: Final index -1?
		if ((*x_in)->elt[(*x_in)->dimSize - 1].index != -1)
			throw LVException(__FILE__, __LINE__, "The index of the last element of the feature vector needs to be -1 (liblinear_predict_probability).");

		// Convert LVsvm_model to svm_model
		auto mdl = std::make_unique<model>();
		LVConvertModel(*model_in, *mdl);

		// Check probability model
		int valid_probability = check_probability_model(mdl.get());
		if (!valid_probability)
			throw LVException(__FILE__, __LINE__, "The selected solver type does not support probability output.");

		// Allocate room for probability estimates
		LVResizeNumericArrayHandle(prob_estimates_out, mdl->nr_class);
		(*prob_estimates_out)->dimSize = mdl->nr_class;

		double highest_prob_label = predict_probability(mdl.get(), reinterpret_cast<feature_node*>((*x_in)->elt), (*prob_estimates_out)->elt);

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

void LVConvertParameter(const LVlinear_parameter &param_in, parameter &param_out){
	param_out.solver_type = param_in.solver_type;
	param_out.eps = param_in.eps;
	param_out.C = param_in.C;
	param_out.p = param_in.p;
	param_out.init_sol = nullptr; // TODO: add support for warm-start

	// Weight label
	if (param_in.weight_label != nullptr && param_in.weight != nullptr){
		if ((*(param_in.weight))->dimSize != (*(param_in.weight_label))->dimSize)
			throw LVException(__FILE__, __LINE__, "Parameter error: Number of elements in weight_label and weight does not match.");

		if ((*(param_in.weight_label))->dimSize > 0)
			param_out.weight_label = (*(param_in.weight_label))->elt;
		else
			param_out.weight_label = nullptr;

		// Weight
		if ((*(param_in.weight))->dimSize > 0){
			param_out.weight = (*(param_in.weight))->elt;
			param_out.nr_weight = (*(param_in.weight))->dimSize;
		}
		else{
			param_out.weight = nullptr;
			param_out.nr_weight = 0;
		}
	}
}

void LVConvertModel(const LVlinear_model &model_in, model &model_out){
	// Assign the parameters
	LVConvertParameter(model_in.param, model_out.param);

	// Copy assignments
	model_out.nr_class = model_in.nr_class;
	model_out.nr_feature = model_in.nr_feature;
	model_out.bias = model_in.bias;

	// w
	if ((*(model_in.w))->dimSize > 0)
		model_out.w = (*(model_in.w))->elt;
	else
		model_out.w = nullptr;

	// label
	if ((*(model_in.label))->dimSize > 0)
		model_out.label = (*(model_in.label))->elt;
	else
		model_out.label = nullptr;
}

void LVConvertModel(const model &model_in, LVlinear_model &model_out){
	// Convert svm_model to LVsvm_model
	model_out.nr_class = model_in.nr_class;
	model_out.nr_feature = model_in.nr_feature;
	model_out.bias = model_in.bias;
	int nr_class = model_in.nr_class;
	int nr_feature = model_in.nr_feature;

	// Label
	if (model_in.label != nullptr){
		LVResizeNumericArrayHandle(model_out.label, nr_class);
		MoveBlock(model_in.label, (*(model_out.label))->elt, nr_class * sizeof(int32_t));
		(*model_out.label)->dimSize = model_in.nr_class;
	}
	else{
		(*model_out.label)->dimSize = 0;
	}

	// n is equal to nr_feature, incremented if bias is present
	int n = nr_feature;
	if (model_in.bias >= 0)
		n++;

	// nr_w is equal to nr_class with one exception
	int nr_w;
	if (model_in.nr_class == 2 && model_in.param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = model_in.nr_class;

	if (model_in.w != nullptr){
		LVResizeNumericArrayHandle(model_out.w, n*nr_w);
		MoveBlock(model_in.w, (*(model_out.w))->elt, nr_w * n * sizeof(double));
		(*model_out.w)->dimSize = nr_w*n;
	}
	else{
		(*model_out.w)->dimSize = 0;
	}
}
