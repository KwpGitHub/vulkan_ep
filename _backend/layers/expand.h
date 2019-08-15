#ifndef EXPAND_H
#define EXPAND_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimension must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.

input: Input tensor
input: A 1-D tensor indicates the shape you want to expand to, following the broadcast rule
output: Output tensor
//*/
//Expand
//INPUTS:                   input_input, shape_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Expand : public Layer {
        typedef struct {
            
			
            Shape_t input_input; Shape_t shape_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        
        std::string input_input; std::string shape_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Expand(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string shape_input, std::string output_output); 

        ~Expand() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Expand::Expand(std::string n) : Layer(n) { }
       
    vuh::Device* Expand::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Expand::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
  		binding.shape_input = tensor_dict[shape_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 

    }
    
    void Expand::call(std::string input_input, std::string shape_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/expand.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[shape_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Expand, Layer>(m, "Expand")
            .def(py::init<std::string> ())
            .def("forward", &Expand::forward)
            .def("init", &Expand::init)
            .def("call", (void (Expand::*) (std::string, std::string, std::string)) &Expand::call);
    }
}

#endif

/* PYTHON STUFF

*/

