#include "../layer.h"
#ifndef LPNORMALIZATION_H
#define LPNORMALIZATION_H 
/*

Given a matrix, apply Lp-normalization along the provided axis.

input: Input matrix
output: Matrix after normalization
//*/
//LpNormalization
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, p
//OPTIONAL_PARAMETERS_TYPE: int, int

//class stuff
namespace backend {   

    class LpNormalization : public Layer {
        typedef struct {
            int axis; int p;
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int axis; int p;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LpNormalization(std::string n, int axis, int p);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string output_output); 

        ~LpNormalization() {}

    };
    
}

#endif

