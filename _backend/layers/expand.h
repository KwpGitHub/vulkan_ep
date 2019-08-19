#ifndef EXPAND_H
#define EXPAND_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
*/

//Expand
//INPUTS:                   input_input, shape_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

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
        Expand();
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _input_input, std::string _shape_input, std::string _output_output); 

        ~Expand() {}
    };

    
    void init_layer_Expand(py::module& m) {
        // py::class_(m, "Expand");
    }
    

}


#endif

