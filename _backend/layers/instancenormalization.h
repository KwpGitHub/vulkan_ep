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
//INPUTS:                   input_i, scale_i, B_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
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
			
            Shape_t input_i; Shape_t scale_i; Shape_t B_i;
            
            Shape_t output_o;
            
        } binding_descriptor;

        float epsilon;
        std::string input_i; std::string scale_i; std::string B_i;
        
        std::string output_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        InstanceNormalization(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( float _epsilon); 
        virtual void bind(std::string _input_i, std::string _scale_i, std::string _B_i, std::string _output_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/instancenormalization.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[scale_i]->data(), *tensor_dict[B_i]->data(), *tensor_dict[output_o]->data());
        }

        ~InstanceNormalization() {}
    };
   
}
#endif

