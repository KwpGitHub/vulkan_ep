#ifndef RANDOMNORMALLIKE_H
#define RANDOMNORMALLIKE_H 

#include "../layer.h"

/*

Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.

input: Input tensor to copy shape and optionally type information from.
output: Output tensor of random values drawn from normal distribution
*/

//RandomNormalLike
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, mean, scale, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float


//class stuff
namespace layers {   

    class RandomNormalLike : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        int dtype; float mean; float scale; float seed;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
       

    public:
        RandomNormalLike(std::string name);
        
        virtual void forward();        
        virtual void init( int _dtype,  float _mean,  float _scale,  float _seed); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~RandomNormalLike() {}
    };
   
}
#endif

