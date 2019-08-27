#ifndef EXPAND_H
#define EXPAND_H 

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
*/

//Expand
//INPUTS:                   input_i, shape_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Expand : public backend::Layer {
        typedef struct {          
            backend::Shape_t input_i; backend::Shape_t shape_i;
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        
        std::string input_i; std::string shape_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Expand(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _input_i, std::string _shape_i, std::string _output_o); 
        virtual void build();

        ~Expand() {}
    };
   
}
#endif

