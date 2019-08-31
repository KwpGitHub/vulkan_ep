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
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> m_shape; int m_dtype; float m_high; float m_low; float m_seed;
        
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        RandomUniform(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _shape,  int _dtype,  float _high,  float _low,  float _seed); 
        virtual void bind(std::string _output_o); 
        virtual void build();

        ~RandomUniform() {}
    };
   
}
#endif

