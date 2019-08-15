#ifndef PRELU_H
#define PRELU_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
input: Input tensor
input: Slope tensor. The shape of slope can be smaller then first input X; if so, its shape must be unidirectional broadcastable to X
output: Output tensor (same size as X)
//*/
//PRelu
//INPUTS:                   X_input, slope_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class PRelu : public Layer {
        typedef struct {
            
			
            Shape_t X_input; Shape_t slope_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        
        std::string X_input; std::string slope_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        PRelu(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string slope_input, std::string Y_output); 

        ~PRelu() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    PRelu::PRelu(std::string n) : Layer(n) { }
       
    vuh::Device* PRelu::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void PRelu::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.slope_input = tensor_dict[slope_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 

    }
    
    void PRelu::call(std::string X_input, std::string slope_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/prelu.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[slope_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<PRelu, Layer>(m, "PRelu")
            .def(py::init<std::string> ())
            .def("forward", &PRelu::forward)
            .def("init", &PRelu::init)
            .def("call", (void (PRelu::*) (std::string, std::string, std::string)) &PRelu::call);
    }
}

#endif

/* PYTHON STUFF

*/

