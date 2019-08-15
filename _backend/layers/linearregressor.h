#ifndef LINEARREGRESSOR_H
#define LINEARREGRESSOR_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Generalized linear regression evaluation.<br>
    If targets is set to 1 (default) then univariate regression is performed.<br>
    If targets is set to M then M sets of coefficients must be passed in as a sequence
    and M results will be output for each input n in N.<br>
    The coefficients array is of length n, and the coefficients for each target are contiguous.
    Intercepts are optional but if provided must match the number of targets.

input: Data to be regressed.
output: Regression outputs (one per target, per example).
//*/
//LinearRegressor
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      coefficients, intercepts, post_transform, targets
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, int, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class LinearRegressor : public Layer {
        typedef struct {
            int post_transform; int targets;
			Shape_t coefficients; Shape_t intercepts;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int post_transform; int targets; std::string coefficients; std::string intercepts;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LinearRegressor(std::string n, int post_transform, int targets);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string coefficients, std::string intercepts, std::string X_input, std::string Y_output); 

        ~LinearRegressor() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    LinearRegressor::LinearRegressor(std::string n, int post_transform, int targets) : Layer(n) { }
       
    vuh::Device* LinearRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void LinearRegressor::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.post_transform = post_transform;
  		binding.targets = targets;
  		binding.coefficients = tensor_dict[coefficients]->shape();
  		binding.intercepts = tensor_dict[intercepts]->shape();
 
    }
    
    void LinearRegressor::call(std::string coefficients, std::string intercepts, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/linearregressor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[coefficients]->data(), *tensor_dict[intercepts]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<LinearRegressor, Layer>(m, "LinearRegressor")
            .def(py::init<std::string, int, int> ())
            .def("forward", &LinearRegressor::forward)
            .def("init", &LinearRegressor::init)
            .def("call", (void (LinearRegressor::*) (std::string, std::string, std::string, std::string)) &LinearRegressor::call);
    }
}

#endif

/* PYTHON STUFF

*/

