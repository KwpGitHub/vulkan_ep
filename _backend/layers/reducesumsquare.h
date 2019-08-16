#include "../layer.h"
#ifndef REDUCESUMSQUARE_H
#define REDUCESUMSQUARE_H 
/*

Computes the sum square of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
input: An input tensor.
output: Reduced output tensor.
//*/
//ReduceSumSquare
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes, keepdims
//OPTIONAL_PARAMETERS_TYPE: Shape_t, int

//class stuff
namespace backend {   

    class ReduceSumSquare : public Layer {
        typedef struct {
            Shape_t axes; int keepdims;
			
            Shape_t data_input;
            
            Shape_t reduced_output;
            
        } binding_descriptor;

        Shape_t axes; int keepdims;
        std::string data_input;
        
        std::string reduced_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ReduceSumSquare(std::string n, Shape_t axes, int keepdims);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string reduced_output); 

        ~ReduceSumSquare() {}

    };
    
}

#endif

