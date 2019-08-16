#include "../layer.h"
#ifndef ARRAYFEATUREEXTRACTOR_H
#define ARRAYFEATUREEXTRACTOR_H 
/*

    Select elements of the input tensor based on the indices passed.<br>
    The indices are applied to the last axes of the tensor.

input: Data to be selected
input: The indices, based on 0 as the first index of any dimension.
output: Selected output data as an array
//*/
//ArrayFeatureExtractor
//INPUTS:                   X_input, Y_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class ArrayFeatureExtractor : public Layer {
        typedef struct {
            
			
            Shape_t X_input; Shape_t Y_input;
            
            Shape_t Z_output;
            
        } binding_descriptor;

        
        std::string X_input; std::string Y_input;
        
        std::string Z_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ArrayFeatureExtractor(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_input, std::string Z_output); 

        ~ArrayFeatureExtractor() {}

    };
    
}

#endif

