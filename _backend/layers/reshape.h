#include "../layer.h"
#ifndef RESHAPE_H
#define RESHAPE_H 
/*

Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor).
input: An input tensor.
input: Specified shape for output.
output: Reshaped data.
//*/
//Reshape
//INPUTS:                   data_input, shape_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reshaped_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Reshape : public Layer {
        typedef struct {
            
			
            Shape_t data_input; Shape_t shape_input;
            
            Shape_t reshaped_output;
            
        } binding_descriptor;

        
        std::string data_input; std::string shape_input;
        
        std::string reshaped_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Reshape(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string shape_input, std::string reshaped_output); 

        ~Reshape() {}

    };
    
}

#endif

