#include "../layer.h"
#ifndef LESS_H
#define LESS_H 
/*

Returns the tensor resulted from performing the `less` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: First input operand for the logical operator.
input: Second input operand for the logical operator.
output: Result tensor.
//*/
//Less
//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   C_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Less : public Layer {
        typedef struct {
            
			
            Shape_t A_input; Shape_t B_input;
            
            Shape_t C_output;
            
        } binding_descriptor;

        
        std::string A_input; std::string B_input;
        
        std::string C_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Less(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string A_input, std::string B_input, std::string C_output); 

        ~Less() {}

    };
    
}

#endif

