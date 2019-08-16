#include "../layer.h"
#ifndef RANDOMUNIFORMLIKE_H
#define RANDOMUNIFORMLIKE_H 
/*

Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.

input: Input tensor to copy shape and optionally type information from.
output: Output tensor of random values drawn from uniform distribution
//*/
//RandomUniformLike
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, high, low, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float

//class stuff
namespace backend {   

    class RandomUniformLike : public Layer {
        typedef struct {
            int dtype; float high; float low; float seed;
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int dtype; float high; float low; float seed;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        RandomUniformLike(std::string n);
    
        void forward() { program->run(); }
        
        void init( int _dtype,  float _high,  float _low,  float _seed); 
        void bind(std::string _input_input, std::string _output_output); 

        ~RandomUniformLike() {}

    };
    
}

#endif

