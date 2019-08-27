#ifndef RANDOMUNIFORMLIKE_H
#define RANDOMUNIFORMLIKE_H 

#include "../layer.h"

/*

Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.

input: Input tensor to copy shape and optionally type information from.
output: Output tensor of random values drawn from uniform distribution
*/

//RandomUniformLike
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, high, low, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float


//class stuff
namespace layers {   

    class RandomUniformLike : public backend::Layer {
        typedef struct {          
            backend::Shape_t input_i;
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        int dtype; float high; float low; float seed;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        RandomUniformLike(std::string name);
        
        virtual void forward();        
        virtual void init( int _dtype,  float _high,  float _low,  float _seed); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~RandomUniformLike() {}
    };
   
}
#endif

