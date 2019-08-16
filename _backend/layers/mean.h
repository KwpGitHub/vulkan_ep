#include "../layer.h"
#ifndef MEAN_H
#define MEAN_H 
/*

Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for mean.
output: Output tensor.
//*/
//Mean
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   mean_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Mean : public Layer {
        typedef struct {
            
			
            
            
            Shape_t mean_output;
            
        } binding_descriptor;

        
        
        
        std::string mean_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Mean(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string mean_output); 

        ~Mean() {}

    };
    
}

#endif

