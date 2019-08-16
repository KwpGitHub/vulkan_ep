#include "../layer.h"
#ifndef SOFTSIGN_H
#define SOFTSIGN_H 
/*

Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.

input: Input tensor
output: The softsign (x/(1+|x|)) values of the input tensor computed element-wise
//*/
//Softsign
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Softsign : public Layer {
        typedef struct {
            
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Softsign(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _input_input, std::string _output_output); 

        ~Softsign() {}

    };
    
}

#endif

