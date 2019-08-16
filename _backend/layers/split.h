#include "../layer.h"
#ifndef SPLIT_H
#define SPLIT_H 
/*
Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.

input: The tensor to split
output: One or more outputs forming list of tensors after splitting
//*/
//Split
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, split
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t

//class stuff
namespace backend {   

    class Split : public Layer {
        typedef struct {
            int axis; Shape_t split;
			
            Shape_t input_input;
            
            
            
        } binding_descriptor;

        int axis; Shape_t split;
        std::string input_input;
        
        
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Split(std::string n);
    
        void forward() { program->run(); }
        
        void init( int _axis,  Shape_t _split); 
        void bind(std::string _input_input); 

        ~Split() {}

    };
    
}

#endif

