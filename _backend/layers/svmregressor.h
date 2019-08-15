#ifndef SVMREGRESSOR_H
#define SVMREGRESSOR_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Support Vector Machine regression prediction and one-class SVM anomaly detection.

input: Data to be regressed.
output: Regression outputs (one score per target per example).
//*/
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
            int kernel_type; int n_supports; int one_class; int post_transform;
			Shape_t coefficients; Shape_t kernel_params; Shape_t rho; Shape_t support_vectors;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int kernel_type; int n_supports; int one_class; int post_transform; std::string coefficients; std::string kernel_params; std::string rho; std::string support_vectors;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        SVMRegressor(std::string n, int kernel_type, int n_supports, int one_class, int post_transform);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string coefficients, std::string kernel_params, std::string rho, std::string support_vectors, std::string X_input, std::string Y_output); 

        ~SVMRegressor() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    SVMRegressor::SVMRegressor(std::string n, int kernel_type, int n_supports, int one_class, int post_transform) : Layer(n) { }
       
    vuh::Device* SVMRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void SVMRegressor::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.kernel_type = kernel_type;
  		binding.n_supports = n_supports;
  		binding.one_class = one_class;
  		binding.post_transform = post_transform;
  		binding.coefficients = tensor_dict[coefficients]->shape();
  		binding.kernel_params = tensor_dict[kernel_params]->shape();
  		binding.rho = tensor_dict[rho]->shape();
  		binding.support_vectors = tensor_dict[support_vectors]->shape();
 
    }
    
    void SVMRegressor::call(std::string coefficients, std::string kernel_params, std::string rho, std::string support_vectors, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/svmregressor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[coefficients]->data(), *tensor_dict[kernel_params]->data(), *tensor_dict[rho]->data(), *tensor_dict[support_vectors]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<SVMRegressor, Layer>(m, "SVMRegressor")
            .def(py::init<std::string, int, int, int, int> ())
            .def("forward", &SVMRegressor::forward)
            .def("init", &SVMRegressor::init)
            .def("call", (void (SVMRegressor::*) (std::string, std::string, std::string, std::string, std::string, std::string)) &SVMRegressor::call);
    }
}

#endif

/* PYTHON STUFF

*/

