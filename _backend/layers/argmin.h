#include "../layer.h"
#ifndef ARGMIN_H
#define ARGMIN_H 
/*

Computes the indices of the min elements of the input tensor's element along the 
provided axis. The resulted tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned. 
The type of the output tensor is integer.
input: An input tensor.
output: Reduced output tensor with integer data type.
//*/
//ArgMin
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, keepdims
//OPTIONAL_PARAMETERS_TYPE: int, int

//class stuff
namespace backend {   

    class ArgMin : public Layer {
        typedef struct {
            int axis; int keepdims;
			
            Shape_t data_input;
            
            Shape_t reduced_output;
            
        } binding_descriptor;

        int axis; int keepdims;
        std::string data_input;
        
        std::string reduced_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ArgMin(std::string n);
    
        void forward() { program->run(); }
        
        void init( int _axis,  int _keepdims); 
        void bind(std::string _data_input, std::string _reduced_output); 

        ~ArgMin() {}

    };
    
}

#endif

