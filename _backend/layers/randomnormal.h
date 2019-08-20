#ifndef RANDOMNORMAL_H
#define RANDOMNORMAL_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.


output: Output tensor of random values drawn from normal distribution
*/

//RandomNormal
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      dtype, mean, scale, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float

//class stuff
namespace backend {   

    class RandomNormal : public Layer {
        typedef struct {
            Shape_t shape; int dtype; float mean; float scale; float seed;
			
            
            
            Shape_t output_output;
            
        } binding_descriptor;

        Shape_t shape; int dtype; float mean; float scale; float seed;
        
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        RandomNormal(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( Shape_t _shape,  int _dtype,  float _mean,  float _scale,  float _seed); 
        void bind(std::string _output_output); 

        ~RandomNormal() {}
    };

}

#endif

