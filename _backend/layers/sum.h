#include "../layer.h"
#ifndef SUM_H
#define SUM_H 
/*

Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for sum.
output: Output tensor.
//*/
//Sum
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   sum_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Sum : public Layer {
        typedef struct {
            
			
            
            
            Shape_t sum_output;
            
        } binding_descriptor;

        
        
        
        std::string sum_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Sum(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string sum_output); 

        ~Sum() {}

    };
    
}

#endif

