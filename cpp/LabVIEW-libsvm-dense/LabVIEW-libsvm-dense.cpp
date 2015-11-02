#include "LabVIEW-libsvm-dense.h"

#include <stdint.h>
#include <exception>
#include <string>
#include <cstring>
#include <memory>
#include <errno.h>
#include <cmath>
#include <climits>

#include <extcode.h>
#include <svm.h>

#include "LVTypeDecl.h"
#include "LVUtility.h"
#include "LVException.h"

// C++14 feature: std::make_unique
// GNU g++-4.9 or later with -std=c++14 enabled is needed on unix (VS2013 has native support)
#if defined(__GNUG__) && (!defined(__cpp_lib_make_unique) || (__cplusplus < __cpp_lib_make_unique))
	#include <make_unique.hpp>
#endif

void LVsvm_train(lvError *lvErr, const LVsvm_problem *prob_in, const LVsvm_parameter *param_in, LVsvm_model * model_out) {
	try {

		// Input verification: Nonempty problem
		if ((*prob_in->x)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Empty problem was passed to svm_train.");

		// Input verification: First inner problem array non-empty (used to define feature vector length).
		if ((*prob_in->x)->elt[0] == nullptr || (*(*prob_in->x)->elt[0])->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "First feature vector in problem is empty.");

		uint32_t n_vectors = (*prob_in->x)->dimSize;
		uint32_t n_features = (*(*prob_in->x)->elt[0])->dimSize;

		// Input validation: Feature vector too large (exceeds max signed int)
		if(n_features > INT_MAX)
			throw LVException(__FILE__, __LINE__, "Feature vector too large (grater than " + std::to_string(INT_MAX) + ")");

		// Input validation: Number of vectors too large (exceeds max signed int)
		if(n_vectors > INT_MAX)
			throw LVException(__FILE__, __LINE__, "Number of vectors too large (grater than " + std::to_string(INT_MAX) + ")");

		// Input verification: Problem dimensions (n_vectors equals n_labels)
		if (n_vectors != (*(prob_in->y))->dimSize)
			throw LVException(__FILE__, __LINE__, "The problem must have an equal number of labels and feature vectors (x and y).");

		// Convert LVsvm_problem to svm_problem
		auto prob = std::make_unique<svm_problem>();
		prob->l = n_vectors;
		prob->y = (*(prob_in->y))->elt;

		// Create node structure (array of arrays is used even though its dense)
		auto x = std::make_unique<svm_node[]>(n_vectors);
		prob->x = x.get();

		for (unsigned int i = 0; i < n_vectors; i++) {
			// Disallow feature vectors of different size, they are truncated in the dot-product anyway.
			if ((*(*(prob_in->x))->elt[i])->dimSize != n_features)
				throw LVException(__FILE__, __LINE__, "Feature vector #" + std::to_string(i) + " differs in length from the rest.");
			
			x[i].dim = n_features;
			x[i].values = (*(*(prob_in->x))->elt[i])->elt;
		}

		// Assign parameters to svm_parameter
		auto param = std::make_unique<svm_parameter>();
		LVConvertParameter(*param_in, *param);

		// Verify parameters
		const char * param_check = svm_check_parameter(prob.get(), param.get());
		if (param_check != nullptr)
			throw LVException(__FILE__, __LINE__, "Parameter check failed with the following error: " + std::string(param_check));

		// Train model
		svm_model *model = svm_train(prob.get(), param.get());

		// Copy the data into LabVIEW memory (hardcopy)
		LVConvertModel(*model, *model_out);

		// Release memory allocated by svm_train
		svm_free_model_content(model);
	}
	catch (LVException &ex) {
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

		ex.returnError(lvErr);
	}
	catch (std::exception &ex) {
		(*(model_out->label))->dimSize = 0;
		(*(model_out->nSV))->dimSize = 0;
		(*(model_out->probA))->dimSize = 0;
		(*(model_out->probB))->dimSize = 0;
		(*(model_out->rho))->dimSize = 0;
		(*(model_out->SV))->dimSize = 0;
		(*(model_out->sv_coef))->dimSize[0] = 0;
		(*(model_out->sv_coef))->dimSize[1] = 0;
		(*(model_out->sv_indices))->dimSize = 0;

		LVException::returnStdException(lvErr, __FILE__, __LINE__, ex);
	}
	catch (...) {
		(*(model_out->label))->dimSize = 0;
		(*(model_out->nSV))->dimSize = 0;
		(*(model_out->probA))->dimSize = 0;
		(*(model_out->probB))->dimSize = 0;
		(*(model_out->rho))->dimSize = 0;
		(*(model_out->SV))->dimSize = 0;
		(*(model_out->sv_coef))->dimSize[0] = 0;
		(*(model_out->sv_coef))->dimSize[1] = 0;
		(*(model_out->sv_indices))->dimSize = 0;

		LVException ex(__FILE__, __LINE__, "Unknown exception has occurred");
		ex.returnError(lvErr);
	}
}

void LVsvm_cross_validation(lvError *lvErr, const LVsvm_problem *prob_in, const LVsvm_parameter *param_in, int32_t nr_fold, LVArray_Hdl<double> target_out) {
	try {

		// Input verification: Nonempty problem
		if ((*prob_in->x)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Empty problem was passed to svm_train.");

		// Input verification: First inner problem array non-empty (used to define feature vector length).
		if ((*prob_in->x)->elt[0] == nullptr || (*(*prob_in->x)->elt[0])->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "First feature vector in problem is empty.");

		uint32_t n_vectors = (*prob_in->x)->dimSize;
		uint32_t n_features = (*(*prob_in->x)->elt[0])->dimSize;

		// Input validation: Feature vector too large (exceeds max signed int)
		if(n_features > INT_MAX)
			throw LVException(__FILE__, __LINE__, "Feature vector too large (grater than " + std::to_string(INT_MAX) + ")");

		// Input validation: Number of vectors too large (exceeds max signed int)
		if(n_vectors > INT_MAX)
			throw LVException(__FILE__, __LINE__, "Number of vectors too large (grater than " + std::to_string(INT_MAX) + ")");

		// Input verification: Problem dimensions
		if (n_vectors != (*(prob_in->y))->dimSize)
			throw LVException(__FILE__, __LINE__, "The problem must have an equal number of labels and feature vectors (x and y).");

		// Convert LVsvm_problem to svm_problem
		auto prob = std::make_unique<svm_problem>();
		prob->l = n_vectors; // The number of nodes
		prob->y = (*(prob_in->y))->elt;

		// Create node structure (array of arrays is used even though its dense)
		auto x = std::make_unique<svm_node[]>(n_vectors);
		prob->x = x.get();

		for (unsigned int i = 0; i < n_vectors; i++) {
			// Disallow feature vectors of different size, they are truncated in the dot-product anyway.
			if ((*(*(prob_in->x))->elt[i])->dimSize != n_features)
				throw LVException(__FILE__, __LINE__, "Feature vector #" + std::to_string(i) + " differs in length from the rest.");

			x[i].dim = static_cast<int>(n_features);
			x[i].values = (*(*(prob_in->x))->elt[i])->elt;
		}

		// Assign parameters to svm_parameter
		auto param = std::make_unique<svm_parameter>();
		LVConvertParameter(*param_in, *param);

		// Verify parameters
		const char * param_check = svm_check_parameter(prob.get(), param.get());
		if (param_check != nullptr)
			throw LVException(__FILE__, __LINE__, "Parameter check failed with the following error: " + std::string(param_check));

		// Allocate room in target_out
		LVResizeNumericArrayHandle(target_out, n_vectors);

		svm_cross_validation(prob.get(), param.get(), nr_fold, (*target_out)->elt);

		(*target_out)->dimSize = n_vectors;
	}
	catch (LVException &ex) {
		(*target_out)->dimSize = 0;
		ex.returnError(lvErr);
	}
	catch (std::exception &ex) {
		(*target_out)->dimSize = 0;
		LVException::returnStdException(lvErr, __FILE__, __LINE__, ex);
	}
	catch (...) {
		(*target_out)->dimSize = 0;
		LVException ex(__FILE__, __LINE__, "Unknown exception has occurred");
		ex.returnError(lvErr);
	}
}

double	LVsvm_predict(lvError *lvErr, const struct LVsvm_model *model_in, const LVArray_Hdl<double> x_in) {
	try {
		// Input validation: Uninitialized model
		if (model_in == nullptr || model_in->SV == nullptr || (*model_in->SV)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Uninitialized model passed to libsvmdense_predict.");

		// Input validation: Empty feature vector
		if (x_in == nullptr || (*x_in)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Empty feature vector passed to libsvmdense_predict.");

		// Input validation: Feature vector too large (exceeds max signed int)
		if((*x_in)->dimSize > INT_MAX)
			throw LVException(__FILE__, __LINE__, "Feature vector too large (grater than " + std::to_string(INT_MAX) + ")");

		// Convert LVsvm_model to svm_model
		auto model = std::make_unique<svm_model>();
		std::unique_ptr<svm_node[]> SV;
		std::unique_ptr<double*[]> sv_coef;
		LVConvertModel(*model_in, *model, SV, sv_coef);

		svm_node node = { static_cast<int>((*x_in)->dimSize), (*x_in)->elt };
		double label = svm_predict(model.get(), &node);

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

double	LVsvm_predict_values(lvError *lvErr, const LVsvm_model *model_in, const LVArray_Hdl<double> x_in, LVArray_Hdl<double> dec_values_out) {
	try {
		// Input validation: Uninitialized model
		if (model_in == nullptr || model_in->SV == nullptr || (*model_in->SV)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Uninitialized model passed to libsvmdense_predict_values.");

		// Input validation: Empty feature vector
		if (x_in == nullptr || (*x_in)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Empty feature vector passed to libsvmdense_predict_values.");

		// Input validation: Feature vector too large (exceeds max signed int)
		if((*x_in)->dimSize > INT_MAX)
			throw LVException(__FILE__, __LINE__, "Feature vector too large (grater than " + std::to_string(INT_MAX) + ")");

		// Convert LVsvm_model to svm_model
		auto model = std::make_unique<svm_model>();
		std::unique_ptr<svm_node[]> SV;
		std::unique_ptr<double*[]> sv_coef;
		LVConvertModel(*model_in, *model, SV, sv_coef);

		// Allocate room for dec_values output
		if (model->param.svm_type == ONE_CLASS ||
			model->param.svm_type == EPSILON_SVR ||
			model->param.svm_type == NU_SVR) {
			LVResizeNumericArrayHandle(dec_values_out, 1);
			(*dec_values_out)->dimSize = 1;
		}
		else {
			size_t n_pairs = model->nr_class * (model->nr_class - 1) / 2;
			LVResizeNumericArrayHandle(dec_values_out, n_pairs);
			(*dec_values_out)->dimSize = static_cast<uint32_t>(n_pairs);
		}

		svm_node node = { static_cast<int>((*x_in)->dimSize), (*x_in)->elt };
		double predicted_label = svm_predict_values(model.get(), &node, (*dec_values_out)->elt);

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

double	LVsvm_predict_probability(lvError *lvErr, const LVsvm_model *model_in, const LVArray_Hdl<double> x_in, LVArray_Hdl<double> prob_estimates_out) {
	try {
		// Input validation: Uninitialized model
		if (model_in == nullptr || model_in->SV == nullptr || (*model_in->SV)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Uninitialized model passed to libsvmdense_predict_probability.");

		// Input validation: Empty feature vector
		if (x_in == nullptr || (*x_in)->dimSize == 0)
			throw LVException(__FILE__, __LINE__, "Empty feature vector passed to libsvmdense_predict_probability.");

		// Input validation: Feature vector too large (exceeds max signed int)
		if((*x_in)->dimSize > INT_MAX)
			throw LVException(__FILE__, __LINE__, "Feature vector too large (grater than " + std::to_string(INT_MAX) + ")");

		// Convert LVsvm_model to svm_model
		auto model = std::make_unique<svm_model>();
		std::unique_ptr<svm_node[]> SV;
		std::unique_ptr<double*[]> sv_coef;
		LVConvertModel(*model_in, *model, SV, sv_coef);

		// Check probability model
		int valid_probability = svm_check_probability_model(model.get());

		if (!valid_probability)
			throw LVException(__FILE__, __LINE__, "The probability model is not valid.");

		// Allocate room for probability estimates
		// Regression and one-class SVM does not modify this value (returns the same as svm_predict)
		if (model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) {
			LVResizeNumericArrayHandle(prob_estimates_out, model->nr_class);
			(*prob_estimates_out)->dimSize = model->nr_class;
		}
		else {
			(*prob_estimates_out)->dimSize = 0;
		}

		svm_node node = { static_cast<int>((*x_in)->dimSize), (*x_in)->elt };
		double highest_prob_label = svm_predict_probability(model.get(), &node, (*prob_estimates_out)->elt);

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

//
// -- Helper functions
//

void LVConvertParameter(const LVsvm_parameter &param_in, svm_parameter &param_out) {
	//-- Copy assignments
	param_out.svm_type = param_in.svm_type;
	param_out.kernel_type = param_in.kernel_type;
	param_out.degree = param_in.degree;
	param_out.gamma = param_in.gamma;
	param_out.coef0 = param_in.coef0;
	param_out.cache_size = param_in.cache_size;
	param_out.eps = param_in.eps;
	param_out.C = param_in.C;
	param_out.nu = param_in.nu;
	param_out.p = param_in.p;
	param_out.shrinking = param_in.shrinking;
	param_out.probability = param_in.probability;

	if ((*param_in.weight)->dimSize != (*param_in.weight_label)->dimSize)
		throw LVException(__FILE__, __LINE__, "Parameter error: the number of elements in weight and weight_label does not match.");

	param_out.nr_weight = (*(param_in.weight))->dimSize;

	//-- Array assigments

	// Weight label
	if (param_in.weight_label != nullptr && (*param_in.weight_label)->dimSize > 0)
		param_out.weight_label = (*param_in.weight_label)->elt;
	else
		param_out.weight_label = nullptr;

	// Weight
	if (param_in.weight != nullptr && (*(param_in.weight))->dimSize > 0)
		param_out.weight = (*(param_in.weight))->elt;
	else
		param_out.weight = nullptr;
}

void LVConvertParameter(const svm_parameter &param_in, LVsvm_parameter &param_out) {
	param_out.svm_type = param_in.svm_type;
	param_out.kernel_type = param_in.kernel_type;
	param_out.degree = param_in.degree;
	param_out.gamma = param_in.gamma;
	param_out.coef0 = param_in.coef0;
	param_out.cache_size = param_in.cache_size;
	param_out.eps = param_in.eps;
	param_out.C = param_in.C;
	param_out.nu = param_in.nu;
	param_out.p = param_in.p;
	param_out.shrinking = param_in.shrinking;
	param_out.probability = param_in.probability;

	int nr_weight = param_in.nr_weight;
	// Weights
	if (nr_weight > 0) {
		if (param_in.weight != nullptr) {
			LVResizeNumericArrayHandle(param_out.weight, nr_weight);
			MoveBlock(param_in.weight, (*(param_out.weight))->elt, nr_weight * sizeof(double));
			(*(param_out.weight))->dimSize = nr_weight;
		}

		if (param_in.weight_label != nullptr) {
			// Weight_label (number of support vectors for each class)
			LVResizeNumericArrayHandle(param_out.weight_label, nr_weight);
			MoveBlock(param_in.weight_label, (*(param_out.weight_label))->elt, nr_weight * sizeof(int32_t));
			(*(param_out.weight_label))->dimSize = nr_weight;
		}
	}
}

void LVConvertModel(const LVsvm_model &model_in, svm_model &model_out, std::unique_ptr<svm_node[]> &SV, std::unique_ptr<double*[]> &sv_coef) {
	// Input verification: Reject uninitialized models from LabVIEW
	if (model_in.l <= 0 || model_in.nSV == nullptr || (*model_in.nSV)->dimSize == 0 || model_in.SV == nullptr || (*model_in.SV)->dimSize == 0)
		throw LVException(__FILE__, __LINE__, "Uninitialized model passed to libsvm.");
	
	// Assign the parameters
	LVConvertParameter(model_in.param, model_out.param);

	//-- Copy assignments
	model_out.nr_class = model_in.nr_class;
	model_out.l = model_in.l;
	model_out.free_sv = 0;

	//-- 1D Array assignments
	// Rho
	if ((*(model_in.rho))->dimSize > 0)
		model_out.rho = (*(model_in.rho))->elt;
	else
		model_out.rho = nullptr;

	// ProbA
	if ((*(model_in.probA))->dimSize > 0)
		model_out.probA = (*(model_in.probA))->elt;
	else
		model_out.probA = nullptr;

	// ProbB
	if ((*(model_in.probB))->dimSize > 0)
		model_out.probB = (*(model_in.probB))->elt;
	else
		model_out.probB = nullptr;

	// sv indices
	if ((*(model_in.sv_indices))->dimSize > 0)
		model_out.sv_indices = (*(model_in.sv_indices))->elt;
	else
		model_out.sv_indices = nullptr;

	// label
	if ((*(model_in.label))->dimSize > 0)
		model_out.label = (*(model_in.label))->elt;
	else
		model_out.label = nullptr;

	// nSV
	if ((*(model_in.nSV))->dimSize > 0)
		model_out.nSV = (*(model_in.nSV))->elt;
	else
		model_out.nSV = nullptr;

	//-- 2D Array assigments (pointer-to-pointer)

	// SV
	if ((*(model_in.SV))->dimSize > 0) {
		uint32_t n_SV = (*(model_in.SV))->dimSize;

		SV = std::make_unique<svm_node[]>(n_SV);
		for (uint32_t i = 0; i < n_SV; i++) {
			SV[i].dim = (*(*model_in.SV)->elt[i])->dimSize;
			SV[i].values = (*(*model_in.SV)->elt[i])->elt;
		}
		model_out.SV = SV.get();
	}
	else {
		model_out.SV = nullptr;
	}

	// sv_coef
	if ((*(model_in.sv_coef))->dimSize > 0) {
		uint32_t *nsv_coef = (*(model_in.sv_coef))->dimSize;
		sv_coef = std::make_unique<double*[]>(nsv_coef[0]);
		for (uint32_t i = 0; i < nsv_coef[0]; i++) {
			sv_coef[i] = &(*(model_in.sv_coef))->elt[i * nsv_coef[1]];
		}
		model_out.sv_coef = sv_coef.get();
	}
	else {
		model_out.sv_coef = nullptr;
	}
}

void LVConvertModel(const svm_model &model_in, LVsvm_model &model_out) {
	// Convert parameters
	LVConvertParameter(model_in.param, model_out.param);

	// Convert svm_model to LVsvm_model
	model_out.nr_class = model_in.nr_class;
	model_out.l = model_in.l;

	int n_class = model_in.nr_class;				// Number of classes
	int n_pairs = n_class*(n_class - 1) / 2;		// Total pairwise count
	int n_SV = model_in.l;							// Total SV count (not to be confused with the nSV member)

	// Label
	if (model_in.label != nullptr) {
		LVResizeNumericArrayHandle(model_out.label, n_class);
		MoveBlock(model_in.label, (*(model_out.label))->elt, n_class * sizeof(int32_t));
		(*model_out.label)->dimSize = model_in.nr_class;
	}

	// Rho
	if (model_in.rho != nullptr) {
		LVResizeNumericArrayHandle(model_out.rho, n_pairs);
		MoveBlock(model_in.rho, (*(model_out.rho))->elt, n_pairs * sizeof(double));
		(*model_out.rho)->dimSize = n_pairs;
	}

	// Probability (probA and probB)
	if ((model_in.param).probability) {
		if (model_in.probA != nullptr) {
			LVResizeNumericArrayHandle(model_out.probA, n_pairs);
			MoveBlock(model_in.probA, (*(model_out.probA))->elt, n_pairs * sizeof(double));
			(*(model_out.probA))->dimSize = n_pairs;
		}

		if (model_in.probB != nullptr) {
			LVResizeNumericArrayHandle(model_out.probB, n_pairs);
			MoveBlock(model_in.probB, (*(model_out.probB))->elt, n_pairs * sizeof(double));
			(*(model_out.probB))->dimSize = n_pairs;
		}
	}
	else {
		if (DSCheckHandle(model_out.probA) != noErr)
			(*(model_out.probA))->dimSize = 0;
		if (DSCheckHandle(model_out.probB) != noErr)
			(*(model_out.probB))->dimSize = 0;
	}

	// nSVs (number of support vectors for each class)
	if (model_in.nSV != nullptr) {
		LVResizeNumericArrayHandle(model_out.nSV, n_class);
		for (int i = 0; i < n_class; i++)
			(*(model_out.nSV))->elt[i] = model_in.nSV[i];
		(*model_out.nSV)->dimSize = n_class;
	}

	// sv_indices
	if (model_in.sv_indices != nullptr) {
		LVResizeNumericArrayHandle(model_out.sv_indices, n_SV);
		MoveBlock(model_in.sv_indices, (*(model_out.sv_indices))->elt, n_SV * sizeof(int32_t));
		(*model_out.sv_indices)->dimSize = n_SV;
	}

	// SV (support vectors) rows: n_SV, cols: n_features
	if (model_in.SV != nullptr && n_SV > 0) {
		int n_features = model_in.SV[0].dim;
		LVResizeHandleArrayHandle(model_out.SV, n_SV);

		for (int i = 0; i < n_SV; i++) {
			// Dense LabVIEW implementation does not allow for feature vectors of different size
			if (model_in.SV[i].dim == n_features){
				if (model_in.SV[i].values != nullptr && model_in.SV[i].dim > 0) {
					LVResizeNumericArrayHandle((*model_out.SV)->elt[i], n_features);

					// Copy data over
					MoveBlock(model_in.SV[i].values, (*(*model_out.SV)->elt[i])->elt, n_features*sizeof(double));

					(*(*model_out.SV)->elt[i])->dimSize = n_features;
				}
				else {
					throw LVException(__FILE__, __LINE__, "Model error: A support vector in the model is invalid (null).");
				}
			}
			else{
				throw LVException(__FILE__, __LINE__, "All support vectors in the model must have same length (libsvm-dense only).");
			}
		}

		(*model_out.SV)->dimSize = n_SV;
	}
	else {
		if (*(model_out.SV) != nullptr) {
			(*model_out.SV)->dimSize = 0;
		}
	}

	// sv_coef
	if (model_in.sv_coef != nullptr) {
		LVResizeNumericArrayHandle(model_out.sv_coef, (n_class - 1) * n_SV);
		for (int i = 0; i < n_class - 1; i++) {
			MoveBlock(model_in.sv_coef[i], (*(model_out.sv_coef))->elt + i*n_SV, n_SV*sizeof(double));
		}

		(*(model_out.sv_coef))->dimSize[0] = n_class - 1;
		(*(model_out.sv_coef))->dimSize[1] = n_SV;
	}
}

//-- Print function (console logging)
void LVsvm_print_function(const char * message) {
	LVUserEventRef *usrEv = loggingUsrEv;
	if (usrEv != nullptr && message != nullptr) {
		// Filter out the progress messages (..... and *)
		if (strcmp(message, ".") != 0 && strcmp(message, "*") != 0) {
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


//
//-- File saving/loading
//

void LVsvm_save_model(lvError *lvErr, const char *path_in, const LVsvm_model *model_in){
	try{
        errno = 0;

		// Convert LVsvm_model to svm_model
		auto model = std::make_unique<svm_model>();
		std::unique_ptr<svm_node[]> SV;
		std::unique_ptr<double*[]> sv_coef;
		LVConvertModel(*model_in, *model, SV, sv_coef);

		int err = svm_save_model(path_in, model.get());

		if (err == -1){
            // Allocate room for output error message (truncated if buffer is too small)
            const size_t bufSz = 256;
            char buf[bufSz] = "";
            std::string errstr;

		#if defined(_WIN32) || defined(_WIN64)
			if(strerror_s(buf, bufSz, errno) != 0)
                errstr = buf;
            else
                errstr = "Unknown error";
        #elif (_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && ! _GNU_SOURCE
            if(strerror_r(errno, buf, bufSz) != 0)
                errstr = buf;
            else
                errstr = "Unknown error";
        #else
            char* gnuerr = strerror_r(errno, buf, bufSz);
            if(gnuerr != nullptr)
                errstr = gnuerr;
            else
                errstr = "Unknown error";
        #endif

        errno = 0;

        throw LVException(__FILE__, __LINE__, "Model load operation failed (" + errstr + ").");
		}
	}
	catch (LVException &ex) {
		ex.returnError(lvErr);
	}
	catch (std::exception &ex) {
		LVException::returnStdException(lvErr, __FILE__, __LINE__, ex);
	}
	catch (...) {
		LVException ex(__FILE__, __LINE__, "Unknown exception has occurred");
		ex.returnError(lvErr);
	}
}

void LVsvm_load_model(lvError *lvErr, const char *path_in, LVsvm_model *model_out){
	try{
        errno = 0;

		svm_model *model = svm_load_model(path_in);

		if (model == nullptr){
            // Allocate room for output error message (truncated if buffer is too small)
            const size_t bufSz = 256;
            char buf[bufSz] = "";
            std::string errstr;

		#if defined(_WIN32) || defined(_WIN64)
			if(strerror_s(buf, bufSz, errno) != 0)
                errstr = buf;
            else
                errstr = "Unknown error";
        #elif (_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && ! _GNU_SOURCE
            if(strerror_r(errno, buf, bufSz) != 0)
                errstr = buf;
            else
                errstr = "Unknown error";
        #else
            char* gnuerr = strerror_r(errno, buf, bufSz);
            if(gnuerr != nullptr)
                errstr = gnuerr;
            else
                errstr = "Unknown error";
        #endif

        errno = 0;

        throw LVException(__FILE__, __LINE__, "Model load operation failed (" + errstr + ").");
		}
		else{
            // libsvm returns uninitialized values for the parameters
            (model->param) = { 0 };

			LVConvertModel(*model, *model_out);
			svm_free_model_content(model);
		}
	}
	catch (LVException &ex) {
		(*(model_out->label))->dimSize = 0;
		(*(model_out->nSV))->dimSize = 0;
		(*(model_out->probA))->dimSize = 0;
		(*(model_out->probB))->dimSize = 0;
		(*(model_out->rho))->dimSize = 0;
		(*(model_out->SV))->dimSize = 0;
		(*(model_out->sv_coef))->dimSize[0] = 0;
		(*(model_out->sv_coef))->dimSize[1] = 0;
		(*(model_out->sv_indices))->dimSize = 0;

		ex.returnError(lvErr);
	}
	catch (std::exception &ex) {
		(*(model_out->label))->dimSize = 0;
		(*(model_out->nSV))->dimSize = 0;
		(*(model_out->probA))->dimSize = 0;
		(*(model_out->probB))->dimSize = 0;
		(*(model_out->rho))->dimSize = 0;
		(*(model_out->SV))->dimSize = 0;
		(*(model_out->sv_coef))->dimSize[0] = 0;
		(*(model_out->sv_coef))->dimSize[1] = 0;
		(*(model_out->sv_indices))->dimSize = 0;

		LVException::returnStdException(lvErr, __FILE__, __LINE__, ex);
	}
	catch (...) {
		(*(model_out->label))->dimSize = 0;
		(*(model_out->nSV))->dimSize = 0;
		(*(model_out->probA))->dimSize = 0;
		(*(model_out->probB))->dimSize = 0;
		(*(model_out->rho))->dimSize = 0;
		(*(model_out->SV))->dimSize = 0;
		(*(model_out->sv_coef))->dimSize[0] = 0;
		(*(model_out->sv_coef))->dimSize[1] = 0;
		(*(model_out->sv_indices))->dimSize = 0;

		LVException ex(__FILE__, __LINE__, "Unknown exception has occurred");
		ex.returnError(lvErr);
	}
}
