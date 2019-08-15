#ifndef SVMREGRESSOR_H
#define SVMREGRESSOR_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Support Vector Machine regression prediction and one-class SVM anomaly detection.

input: Data to be regressed.
output: Regression outputs (one score per target per example).

*/
//SVMRegressor
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      coefficients, kernel_params, kernel_type, n_supports, one_class, post_transform, rho, support_vectors
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, int, int, int, int, Tensor*, Tensor*

namespace py = pybind11;

//class stuff
namespace backend {   

    class SVMRegressor : public Layer {
        typedef struct {    
            Tensor* coefficients; Tensor* kernel_params; int kernel_type; int n_supports; int one_class; int post_transform; Tensor* rho; Tensor* support_vectors;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            int kernel_type; int n_supports; int one_class; int post_transform;
		Shape_t coefficients; Shape_t kernel_params; Shape_t rho; Shape_t support_vectors;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        SVMRegressor(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~SVMRegressor() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    SVMRegressor::SVMRegressor(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/svmregressor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* SVMRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void SVMRegressor::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.kernel_type = parameters.kernel_type;
  		binding.n_supports = parameters.n_supports;
  		binding.one_class = parameters.one_class;
  		binding.post_transform = parameters.post_transform;
  		binding.coefficients = parameters.coefficients->shape();
  		binding.kernel_params = parameters.kernel_params->shape();
  		binding.rho = parameters.rho->shape();
  		binding.support_vectors = parameters.support_vectors->shape();
 
        program->bind(binding, *parameters.coefficients->data(), *parameters.kernel_params->data(), *parameters.rho->data(), *parameters.support_vectors->data(), *input.X_input->data(), *output.Y_output->data());
    }
    
    void SVMRegressor::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<SVMRegressor, Layer>(m, "SVMRegressor")
            .def("forward", &SVMRegressor::forward);    
    }
}*/

#endif
