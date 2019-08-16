#include "../layer.h"
#ifndef THRESHOLDEDRELU_H
#define THRESHOLDEDRELU_H 
/*

ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.

input: Input tensor
output: Output tensor
//*/
//ThresholdedRelu
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

    class ThresholdedRelu : public Layer {
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
        ThresholdedRelu(std::string n, float alpha);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~ThresholdedRelu() {}

    };
    
}

#endif

