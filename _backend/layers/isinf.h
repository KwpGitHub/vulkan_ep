#include "../layer.h"
#ifndef ISINF_H
#define ISINF_H 
/*
Map infinity to true and other values to false.
input: input
output: output
//*/
//IsInf
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      detect_negative, detect_positive
//OPTIONAL_PARAMETERS_TYPE: int, int

//class stuff
namespace backend {   

    class IsInf : public Layer {
        typedef struct {
            int detect_negative; int detect_positive;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int detect_negative; int detect_positive;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        IsInf(std::string n, int detect_negative, int detect_positive);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~IsInf() {}

    };
    
}

#endif

