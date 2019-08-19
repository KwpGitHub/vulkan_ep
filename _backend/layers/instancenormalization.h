#ifndef INSTANCENORMALIZATION_H
#define INSTANCENORMALIZATION_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.


input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
input: The input 1-dimensional scale tensor of size C.
input: The input 1-dimensional bias tensor of size C.
output: The output tensor of the same shape as input.
*/

//InstanceNormalization
//INPUTS:                   input_input, scale_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      epsilon
//OPTIONAL_PARAMETERS_TYPE: float

//class stuff
namespace backend {   

    class InstanceNormalization : public Layer {
        typedef struct {
            float epsilon;
			
            Shape_t input_input; Shape_t scale_input; Shape_t B_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        float epsilon;
        std::string input_input; std::string scale_input; std::string B_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        InstanceNormalization();
    
        void forward() { program->run(); }
        
        void init( float _epsilon); 
        void bind(std::string _input_input, std::string _scale_input, std::string _B_input, std::string _output_output); 

        ~InstanceNormalization() {}
    };

    
    void init_layer_InstanceNormalization(py::module& m) {
        // py::class_(m, "InstanceNormalization");
    }
    

}


#endif

