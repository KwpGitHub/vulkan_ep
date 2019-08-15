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

*/
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
            Tensor* coefficients; Tensor* intercepts; int post_transform; int targets;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            int post_transform; int targets;
		Shape_t coefficients; Shape_t intercepts;
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
        LinearRegressor(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~LinearRegressor() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    LinearRegressor::LinearRegressor(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/linearregressor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* LinearRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void LinearRegressor::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.post_transform = parameters.post_transform;
  		binding.targets = parameters.targets;
  		binding.coefficients = parameters.coefficients->shape();
  		binding.intercepts = parameters.intercepts->shape();
 
        program->bind(binding, *parameters.coefficients->data(), *parameters.intercepts->data(), *input.X_input->data(), *output.Y_output->data());
    }
    
    void LinearRegressor::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<LinearRegressor, Layer>(m, "LinearRegressor")
            .def("forward", &LinearRegressor::forward);    
    }
}*/

#endif
