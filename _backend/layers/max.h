#include "../layer.h"
#ifndef MAX_H
#define MAX_H 
/*

Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for max.
output: Output tensor.
//*/
//Max
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   max_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Max : public Layer {
        typedef struct {
            
			
            
            
            Shape_t max_output;
            
        } binding_descriptor;

        
        
        
        std::string max_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Max(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string max_output); 

        ~Max() {}

    };
    
}

#endif

