#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Support Vector Machine classifier

input: Data to be classified.
output: Classification outputs (one class per example).
output: Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for each class, otherwise they are raw scores.

*/
//SVMClassifier
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output, Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, coefficients, kernel_params, kernel_type, post_transform, prob_a, prob_b, rho, support_vectors, vectors_per_class
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*, Tensor*, int, int, Tensor*, Tensor*, Tensor*, Tensor*, Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class SVMClassifier : public Layer {
        typedef struct {    
            Shape_t classlabels_ints; Tensor* classlabels_strings; Tensor* coefficients; Tensor* kernel_params; int kernel_type; int post_transform; Tensor* prob_a; Tensor* prob_b; Tensor* rho; Tensor* support_vectors; Shape_t vectors_per_class;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output; Tensor* Z_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t classlabels_ints; int kernel_type; int post_transform; Shape_t vectors_per_class;
		Shape_t classlabels_strings; Shape_t coefficients; Shape_t kernel_params; Shape_t prob_a; Shape_t prob_b; Shape_t rho; Shape_t support_vectors;
            Shape_t X_input;
            
            Shape_t Y_output; Shape_t Z_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        SVMClassifier(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~SVMClassifier() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    SVMClassifier::SVMClassifier(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/svmclassifier.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* SVMClassifier::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void SVMClassifier::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
  		binding.Z_output = output.Z_output->shape();
 
		binding.classlabels_ints = parameters.classlabels_ints;
  		binding.kernel_type = parameters.kernel_type;
  		binding.post_transform = parameters.post_transform;
  		binding.vectors_per_class = parameters.vectors_per_class;
  		binding.classlabels_strings = parameters.classlabels_strings->shape();
  		binding.coefficients = parameters.coefficients->shape();
  		binding.kernel_params = parameters.kernel_params->shape();
  		binding.prob_a = parameters.prob_a->shape();
  		binding.prob_b = parameters.prob_b->shape();
  		binding.rho = parameters.rho->shape();
  		binding.support_vectors = parameters.support_vectors->shape();
 
        program->bind(binding, *parameters.classlabels_strings->data(), *parameters.coefficients->data(), *parameters.kernel_params->data(), *parameters.prob_a->data(), *parameters.prob_b->data(), *parameters.rho->data(), *parameters.support_vectors->data(), *input.X_input->data(), *output.Y_output->data(), *output.Z_output->data());
    }
    
    void SVMClassifier::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<SVMClassifier, Layer>(m, "SVMClassifier")
            .def("forward", &SVMClassifier::forward);    
    }
}*/

#endif
