#ifndef RANDOMUNIFORM_H
#define RANDOMUNIFORM_H 

#include "../layer.h"

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
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               shape
//PARAMETER_TYPES:          std::vector<int>
//OPTIONAL_PARAMETERS:      dtype, high, low, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float


//class stuff
namespace layers {   

    class RandomUniform : public backend::Layer {
        typedef struct {          
            
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::vector<int> shape; int dtype; float high; float low; float seed;
        
        
        std::string output_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        RandomUniform(std::string name);
        
        void forward() { program->run(); }
        
        virtual void init( std::vector<int> _shape,  int _dtype,  float _high,  float _low,  float _seed); 
        virtual void bind(std::string _output_o); 
        virtual void build();

        ~RandomUniform() {}
    };
   
}
#endif

