#ifndef INSTANCENORMALIZATION_H
#define INSTANCENORMALIZATION_H 

#include "../layer.h"

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
namespace layers {   

    class InstanceNormalization : public backend::Layer {
        typedef struct {          
            backend::Shape_t input_i; backend::Shape_t scale_i; backend::Shape_t B_i;
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;

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
        virtual void build();

        ~InstanceNormalization() {}
    };
   
}
#endif

