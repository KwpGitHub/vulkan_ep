#pragma once
#ifndef MULTINOMIAL_H
#define MULTINOMIAL_H 

#include "../layer.h"

/*

Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.

input: Input tensor with shape [batch_size, class_size], where class_size is the number of all possible outcomes. Each value along the axis zero represents the unnormalized log-probability of each corresponding outcome in a batch.
output: Output tensor with shape [batch_size, sample_size], where sample_size is the number of times to sample. Each value along the axis zero represents the outcome of the corresponding sample in a batch.
*/

//Multinomial
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, sample_size, seed
//OPTIONAL_PARAMETERS_TYPE: int, int, float


//class stuff
namespace layers {   

    class Multinomial : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        int m_dtype; int m_sample_size; float m_seed;
        std::string m_input_i;
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        Multinomial(std::string name);
        
        virtual void forward();        
        virtual void init( int _dtype,  int _sample_size,  float _seed); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~Multinomial() {}
    };
   
}
#endif

