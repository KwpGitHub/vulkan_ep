#ifndef RANDOMUNIFORMLIKE_H
#define RANDOMUNIFORMLIKE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
namespace backend {   

    class RandomUniformLike : public Layer {
        typedef struct {
            int dtype; float high; float low; float seed;
			
            Shape_t input_i;
            
            Shape_t output_o;
            
        } binding_descriptor;

        int dtype; float high; float low; float seed;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        RandomUniformLike(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( int _dtype,  float _high,  float _low,  float _seed); 
        virtual void bind(std::string _input_i, std::string _output_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomuniformlike.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[output_o]->data());
        }

        ~RandomUniformLike() {}
    };
   
}
#endif

