#ifndef RANDOMUNIFORM_H
#define RANDOMUNIFORM_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.


output: Output tensor of random values drawn from uniform distribution
*/

//RandomUniform
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      dtype, high, low, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float

//class stuff
namespace backend {   

    class RandomUniform : public Layer {
        typedef struct {
            Shape_t shape; int dtype; float high; float low; float seed;
			
            
            
            Shape_t output_output;
            
        } binding_descriptor;

        Shape_t shape; int dtype; float high; float low; float seed;
        
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        RandomUniform(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( Shape_t _shape,  int _dtype,  float _high,  float _low,  float _seed); 
        void bind(std::string _output_output); 

        ~RandomUniform() {}
    };

}

#endif

