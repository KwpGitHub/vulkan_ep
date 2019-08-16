#include "../layer.h"
#ifndef CONSTANT_H
#define CONSTANT_H 
/*
A constant tensor.

output: Output tensor containing the same value of the provided tensor.
//*/
//Constant
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               value
//PARAMETER_TYPES:          Tensor*
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Constant : public Layer {
        typedef struct {
            
			Shape_t value;
            
            
            Shape_t output_output;
            
        } binding_descriptor;

        std::string value;
        
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Constant(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string value, std::string output_output); 

        ~Constant() {}

    };
    
}

#endif

