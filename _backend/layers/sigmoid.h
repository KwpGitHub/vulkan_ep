#include "../layer.h"
#ifndef SIGMOID_H
#define SIGMOID_H 
/*

Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.

input: Input tensor
output: Output tensor
//*/
//Sigmoid
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Sigmoid : public Layer {
        typedef struct {
            
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Sigmoid(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~Sigmoid() {}

    };
    
}

#endif

