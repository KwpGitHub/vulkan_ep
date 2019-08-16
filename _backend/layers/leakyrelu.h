#include "../layer.h"
#ifndef LEAKYRELU_H
#define LEAKYRELU_H 
/*

LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.

input: Input tensor
output: Output tensor
//*/
//LeakyRelu
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha
//OPTIONAL_PARAMETERS_TYPE: float

//class stuff
namespace backend {   

    class LeakyRelu : public Layer {
        typedef struct {
            float alpha;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        float alpha;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LeakyRelu(std::string n, float alpha);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~LeakyRelu() {}

    };
    
}

#endif

