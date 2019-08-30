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
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        float m_epsilon;
        std::string m_input_i; std::string m_scale_i; std::string m_B_i;
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        InstanceNormalization(std::string name);
        
        virtual void forward();        
        virtual void init( float _epsilon); 
        virtual void bind(std::string _input_i, std::string _scale_i, std::string _B_i, std::string _output_o); 
        virtual void build();

        ~InstanceNormalization() {}
    };
   
}
#endif

