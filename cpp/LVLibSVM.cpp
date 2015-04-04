#include "stdafx.h"
#include "LVLibSVM.h"

#include <stdint.h>
#include <exception>
#include <string>
#include <memory>
#include <extcode.h>
#include <svm.h>

#include "LVTypeDecl.h"
#include "LVUtility.h"
#include "LVException.h"




void LVsvm_train(lvError *lvErr, const LVsvm_problem *prob_in, const LVsvm_parameter *param_in, LVsvm_model * model_out){
	try{
		// Convert LVsvm_problem to svm_problem
		std::unique_ptr<svm_problem> prob(new svm_problem);
		prob->l = prob_in->l;
		prob->y = (*(prob_in->y))->elt;

		// Create and array of pointers (sparse datastructure)
		std::unique_ptr<svm_node*[]> x(new svm_node*[prob_in->l]);
		prob->x = x.get();

		auto x_in = prob_in->x;
		for (unsigned int i = 0; i < (*x_in)->dimSize; i++){
			// Assign the innermost svm_node array pointers to the array of pointers
			auto xi_in_Hdl = (*x_in)->elt[i];
			x[i] = reinterpret_cast<svm_node*>( (*xi_in_Hdl)->elt );
		}
		
		// Assign parameters to svm_parameter
		std::unique_ptr<svm_parameter> param(new svm_parameter());
		LVConvertParameter(param_in, param.get());

		// Verify parameters
		const char * param_check = svm_check_parameter(prob.get(), param.get());
		if (param_check != nullptr)
			throw LVException(__FILE__, __LINE__, "Parameter check failed with the following error: " + std::string(param_check));

		// Train model
		svm_model *model = svm_train(prob.get(), param.get());
		
		// Copy the data into LabVIEW memory (hardcopy)
		LVConvertModel(model, model_out);

		// Release memory allocated by svm_train
		svm_free_model_content(model);
	}
	catch (LVException &ex) {
		ex.returnError(lvErr);
		// To avoid LabVIEW reading and utilizing bad memory, the dimension sizes of arrays is set to zero
		(*(model_out->label))->dimSize = 0;
		(*(model_out->nSV))->dimSize = 0;
		(*(model_out->probA))->dimSize = 0;
		(*(model_out->probB))->dimSize = 0;
		(*(model_out->rho))->dimSize = 0;
		(*(model_out->SV))->dimSize = 0;
		(*(model_out->sv_coef))->dimSize[0] = 0;
		(*(model_out->sv_coef))->dimSize[1] = 0;
		(*(model_out->sv_indices))->dimSize = 0;
	}
	catch (std::exception &ex) {
		LVException::returnStdException(lvErr, __FILE__, __LINE__, ex);
		(*(model_out->label))->dimSize = 0;
		(*(model_out->nSV))->dimSize = 0;
		(*(model_out->probA))->dimSize = 0;
		(*(model_out->probB))->dimSize = 0;
		(*(model_out->rho))->dimSize = 0;
		(*(model_out->SV))->dimSize = 0;
		(*(model_out->SV))->dimSize = 0;
		(*(model_out->sv_coef))->dimSize[0] = 0;
		(*(model_out->sv_coef))->dimSize[1] = 0;
		(*(model_out->sv_indices))->dimSize = 0;
	}
	catch (...) {
		LVException ex(__FILE__, __LINE__, "Unknown exception has occurred");
		ex.returnError(lvErr);
		(*(model_out->label))->dimSize = 0;
		(*(model_out->nSV))->dimSize = 0;
		(*(model_out->probA))->dimSize = 0;
		(*(model_out->probB))->dimSize = 0;
		(*(model_out->rho))->dimSize = 0;
		(*(model_out->SV))->dimSize = 0;
		(*(model_out->sv_coef))->dimSize[0] = 0;
		(*(model_out->sv_coef))->dimSize[1] = 0;
		(*(model_out->sv_indices))->dimSize = 0;
	}
}

void LVsvm_cross_validation(lvError *lvErr, const LVsvm_problem *prob_in, const LVsvm_parameter *param_in, int32_t nr_fold, LVArray_Hdl<double> target_out){
	try{
		// Convert LVsvm_problem to svm_problem
		std::unique_ptr<svm_problem> prob(new svm_problem);
		prob->l = prob_in->l;
		prob->y = (*(prob_in->y))->elt;

		// Create and array of pointers (sparse datastructure)
		std::unique_ptr<svm_node*[]> x(new svm_node*[prob_in->l]);
		prob->x = x.get();

		auto x_in = prob_in->x;
		for (unsigned int i = 0; i < (*x_in)->dimSize; i++){
			// Assign the innermost svm_node array pointers to the array of pointers
			auto xi_in_Hdl = (*x_in)->elt[i];
			x[i] = reinterpret_cast<svm_node*>((*xi_in_Hdl)->elt);
		}

		// Assign parameters to svm_parameter
		std::unique_ptr<svm_parameter> param(new svm_parameter());
		LVConvertParameter(param_in, param.get());

		// Verify parameters
		const char * param_check = svm_check_parameter(prob.get(), param.get());
		if (param_check != nullptr)
			throw LVException(__FILE__, __LINE__, "Parameter check failed with the following error: " + std::string(param_check));

		// Allocate room in target_out
		LVResizeNumericArrayHandle(target_out, prob_in->l);

		svm_cross_validation(prob.get(), param.get(), nr_fold, (*target_out)->elt);

		(*target_out)->dimSize = prob_in->l;
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

double	LVsvm_predict(lvError *lvErr, const struct LVsvm_model *model_in, const LVArray_Hdl<LVsvm_node> x_in){
	try{
		// Convert LVsvm_model to svm_model
		std::unique_ptr<svm_model> model(new svm_model);
		std::unique_ptr<svm_node*[]> SV;
		std::unique_ptr<double*[]> sv_coef;
		LVConvertModel(model_in, model.get(), SV, sv_coef);

		double label = svm_predict(model.get(), reinterpret_cast<svm_node*>((*x_in)->elt));

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

double	LVsvm_predict_values(lvError *lvErr, const LVsvm_model *model_in, const LVArray_Hdl<LVsvm_node> x_in, LVArray_Hdl<double> dec_values_out){
	try{
		// Convert LVsvm_model to svm_model
		std::unique_ptr<svm_model> model(new svm_model);
		std::unique_ptr<svm_node*[]> SV;
		std::unique_ptr<double*[]> sv_coef;
		LVConvertModel(model_in, model.get(), SV, sv_coef);

		// Allocate room for dec_values output
		if (model->param.svm_type == ONE_CLASS ||
			model->param.svm_type == EPSILON_SVR ||
			model->param.svm_type == NU_SVR){

			LVResizeNumericArrayHandle(dec_values_out, 1);
			(*dec_values_out)->dimSize = 1;
		}
		else{
			size_t nr_pairs = model->nr_class * (model->nr_class - 1) / 2;
			LVResizeNumericArrayHandle(dec_values_out, nr_pairs);
			(*dec_values_out)->dimSize = static_cast<uint32_t>(nr_pairs);
		}

		double predicted_label = svm_predict_values(model.get(), reinterpret_cast<svm_node*>((*x_in)->elt), (*dec_values_out)->elt);

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

double	LVsvm_predict_probability(lvError *lvErr, const LVsvm_model *model_in, const LVArray_Hdl<LVsvm_node> x_in, LVArray_Hdl<double> prob_estimates_out){
	try{
		// Convert LVsvm_model to svm_model
		std::unique_ptr<svm_model> model(new svm_model);
		std::unique_ptr<svm_node*[]> SV;
		std::unique_ptr<double*[]> sv_coef;
		LVConvertModel(model_in, model.get(), SV, sv_coef);

		// Check probability model
		int valid_probability = svm_check_probability_model(model.get());

		if (!valid_probability)
			throw LVException(__FILE__, __LINE__, "The probability model is not valid.");

		// Allocate room for probability estimates
		// Regression and one-class SVM does not modify this value (returns the same as svm_predict)
		if (model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC){
			LVResizeNumericArrayHandle(prob_estimates_out, model->nr_class);
			(*prob_estimates_out)->dimSize = model->nr_class;
		}
		else {
			(*prob_estimates_out)->dimSize = 0;
		}

		double highest_prob_label = svm_predict_probability(model.get(), reinterpret_cast<svm_node*>((*x_in)->elt), (*prob_estimates_out)->elt);

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

// File operations
//int LVsvm_save_model(lvError *lvErr, const char *model_file_name_in, const svm_model *model_in);
//void LVsvm_load_model(lvError *lvErr, const char *model_file_name_in, svm_model *model_out);


//-- Print function (console logging)
void LVsvm_print_function(const char * message){
	LVUserEventRef *usrEv = loggingUsrEv;
	if (usrEv != nullptr && message != nullptr){
		// Filter out the progress messages (..... and *)
		if (strcmp(message, ".") != 0 && strcmp(message, "*") != 0){
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

void LVsvm_set_logging_userevent(lvError *lvErr, LVUserEventRef *loggingUserEvent_in) { 
	loggingUsrEv = loggingUserEvent_in; 
	svm_set_print_string_function(LVsvm_print_function);
}

void LVsvm_get_logging_userevent(lvError *lvErr, LVUserEventRef *loggingUserEvent_out) { 
	loggingUserEvent_out = loggingUsrEv; 
}

void LVsvm_delete_logging_userevent(lvError *lvErr, LVUserEventRef *loggingUserEvent_out) { 
	loggingUserEvent_out = loggingUsrEv; 
	loggingUsrEv = nullptr;
}


// -- Helper functions

void LVConvertParameter(const LVsvm_parameter *param_in, svm_parameter *param_out){
	//-- Copy assignments

	param_out->svm_type = param_in->svm_type;
	param_out->kernel_type = param_in->kernel_type;
	param_out->degree = param_in->degree;
	param_out->gamma = param_in->gamma;
	param_out->coef0 = param_in->coef0;
	param_out->cache_size = param_in->cache_size;
	param_out->eps = param_in->eps;
	param_out->C = param_in->C;
	param_out->nu = param_in->nu;
	param_out->p = param_in->p;
	param_out->shrinking = param_in->shrinking;
	param_out->probability = param_in->probability;


	if ((*(param_in->weight))->dimSize != (*(param_in->weight_label))->dimSize)
		throw LVException(__FILE__, __LINE__, "Parameter error: the number of elements in weight and weight_label does not match.");
	
	param_out->nr_weight = (*(param_in->weight))->dimSize;

	//-- Array assigments

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


void LVConvertParameter(const svm_parameter *param_in, LVsvm_parameter *param_out){
	param_out->svm_type = param_in->svm_type;
	param_out->kernel_type = param_in->kernel_type;
	param_out->degree = param_in->degree;
	param_out->gamma = param_in->gamma;
	param_out->coef0 = param_in->coef0;
	param_out->cache_size = param_in->cache_size;
	param_out->eps = param_in->eps;
	param_out->C = param_in->C;
	param_out->nu = param_in->nu;
	param_out->p = param_in->p;
	param_out->shrinking = param_in->shrinking;
	param_out->probability = param_in->probability;

	int nr_weight = param_in->nr_weight;

	// Weight
	LVResizeNumericArrayHandle(param_out->weight, nr_weight);
	MoveBlock(param_in->weight, (*(param_out->weight))->elt, nr_weight * sizeof(double));
	(*(param_out->weight))->dimSize = nr_weight;

	// Weight_label (number of support vectors for each class)
	LVResizeNumericArrayHandle(param_out->weight_label, nr_weight);
	MoveBlock(param_in->weight_label, (*(param_out->weight_label))->elt, nr_weight * sizeof(int32_t));
	(*(param_out->weight_label))->dimSize = nr_weight;
}


void LVConvertModel(const LVsvm_model *model_in, svm_model *model_out, std::unique_ptr<svm_node*[]> &SV, std::unique_ptr<double*[]> &sv_coef){
	// Assign the parameters
	LVConvertParameter(&model_in->param, &model_out->param);

	//-- Copy assignments
	model_out->nr_class = model_in->nr_class;
	model_out->l = model_in->l;
	model_out->free_sv = 0;

	//-- 1D Array assignments
	// Rho
	if ((*(model_in->rho))->dimSize > 0)
		model_out->rho = (*(model_in->rho))->elt;
	else
		model_out->rho = nullptr;

	// ProbA
	if ((*(model_in->probA))->dimSize > 0)
		model_out->probA = (*(model_in->probA))->elt;
	else
		model_out->probA = nullptr;

	// ProbB
	if ((*(model_in->probB))->dimSize > 0)
		model_out->probB = (*(model_in->probB))->elt;
	else
		model_out->probB = nullptr;

	// sv indices
	if ((*(model_in->sv_indices))->dimSize > 0)
		model_out->sv_indices = (*(model_in->sv_indices))->elt;
	else
		model_out->sv_indices = nullptr;

	// label
	if ((*(model_in->label))->dimSize > 0)
		model_out->label = (*(model_in->label))->elt;
	else
		model_out->label = nullptr;

	// nSV
	if ((*(model_in->nSV))->dimSize > 0)
		model_out->nSV = (*(model_in->nSV))->elt;
	else
		model_out->nSV = nullptr;


	//-- 2D Array assigments (pointer-to-pointer)

	// SV
	if ((*(model_in->SV))->dimSize > 0){
		uint32_t nSV_in = (*(model_in->SV))->dimSize;
		SV = std::unique_ptr<svm_node*[]>(new svm_node*[nSV_in]);
		for (uint32_t i = 0; i < nSV_in; i++){
			SV[i] = reinterpret_cast<svm_node*>((*(*(model_in->SV))->elt[i])->elt);
		}
		model_out->SV = SV.get();
	}
	else {
		model_out->SV = nullptr;
	}

	// sv_coef
	if ((*(model_in->sv_coef))->dimSize > 0){
		uint32_t *nsv_coef = (*(model_in->sv_coef))->dimSize;
		sv_coef = std::unique_ptr<double*[]>(new double*[nsv_coef[0]]);
		for (uint32_t i = 0; i < nsv_coef[0]; i++){
			sv_coef[i] = &(*(model_in->sv_coef))->elt[i * nsv_coef[1]];
		}
		model_out->sv_coef = sv_coef.get();
	}
	else {
		model_out->sv_coef = nullptr;
	}
}

void LVConvertModel(const svm_model *model_in, LVsvm_model *model_out){
	// Convert parameters
	LVConvertParameter(&model_in->param, &model_out->param);
	
	// Convert svm_model to LVsvm_model
	model_out->nr_class = model_in->nr_class;
	model_out->l = model_in->l;

	int nr_class = model_in->nr_class;				// Number of classes
	int nr_pairs = nr_class*(nr_class - 1) / 2;	// Total pairwise count
	int l = model_in->l;							// Total SV count

	// Label
	LVResizeNumericArrayHandle(model_out->label, nr_class);
	MoveBlock(model_in->label, (*(model_out->label))->elt, nr_class * sizeof(int32_t));
	(*model_out->label)->dimSize = model_in->nr_class;

	// Rho
	LVResizeNumericArrayHandle(model_out->rho, nr_pairs);
	MoveBlock(model_in->rho, (*(model_out->rho))->elt, nr_pairs * sizeof(double));
	(*model_out->rho)->dimSize = nr_pairs;


	// Probability (probA and probB)
	if ((model_in->param).probability){
		LVResizeNumericArrayHandle(model_out->probA, nr_pairs);
		LVResizeNumericArrayHandle(model_out->probB, nr_pairs);

		MoveBlock(model_in->probA, (*(model_out->probA))->elt, nr_pairs * sizeof(double));
		MoveBlock(model_in->probB, (*(model_out->probB))->elt, nr_pairs * sizeof(double));

		(*(model_out->probA))->dimSize = nr_pairs;
		(*(model_out->probB))->dimSize = nr_pairs;
	}
	else{
		if (DSCheckHandle(model_out->probA) != noErr)
			(*(model_out->probA))->dimSize = 0;
		if (DSCheckHandle(model_out->probB) != noErr)
			(*(model_out->probB))->dimSize = 0;
	}

	// nSVs (number of support vectors for each class)
	LVResizeNumericArrayHandle(model_out->nSV, nr_class);
	MoveBlock(model_in->nSV, (*(model_out->nSV))->elt, nr_class * sizeof(int32_t));
	(*model_out->nSV)->dimSize = nr_class;

	// sv_indices
	LVResizeNumericArrayHandle(model_out->sv_indices, l);
	MoveBlock(model_in->sv_indices, (*(model_out->sv_indices))->elt, l * sizeof(int32_t));
	(*model_out->sv_indices)->dimSize = l;


	// SV (support vectors) - total_sv (l) outer dim, variable inner dim
	LVResizeHandleArrayHandle(model_out->SV, l);

	for (int i = 0; i < l; i++){
		// Find inner size by looking for -1 index (except precompute, which has a single element)
		size_t n_nodes = 1;

		const svm_node *p = model_in->SV[i];
		if (model_in->param.kernel_type != PRECOMPUTED){
			while (p->index != -1)
			{
				n_nodes++;
				p++;
			}
		}

		LVResizeCompositeArrayHandle((*(model_out->SV))->elt[i], n_nodes);

		// Copy data over
		MoveBlock(model_in->SV[i], (*(*(model_out->SV))->elt[i])->elt, sizeof(svm_node)*n_nodes);
		(*(*(model_out->SV))->elt[i])->dimSize = static_cast<uint32_t>(n_nodes);
	}

	(*(model_out->SV))->dimSize = l;

	// sv_coef
	LVResizeNumericArrayHandle(model_out->sv_coef, (nr_class - 1) * l);
	for (int i = 0; i < nr_class - 1; i++){
		MoveBlock(model_in->sv_coef[i], (*(model_out->sv_coef))->elt + i*l, l*sizeof(double));
	}

	(*(model_out->sv_coef))->dimSize[0] = nr_class - 1;
	(*(model_out->sv_coef))->dimSize[1] = l;
}
