#ifndef RANDOMNORMAL_H
#define RANDOMNORMAL_H 

#include "../layer.h"

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
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               shape
//PARAMETER_TYPES:          std::vector<int>
//OPTIONAL_PARAMETERS:      dtype, mean, scale, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float


//class stuff
namespace layers {   

    class RandomNormal : public backend::Layer {
        typedef struct {          
            
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;

        std::vector<int> shape; int dtype; float mean; float scale; float seed;
        
        
        std::string output_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        RandomNormal(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( std::vector<int> _shape,  int _dtype,  float _mean,  float _scale,  float _seed); 
        virtual void bind(std::string _output_o); 
        virtual void build();

        ~RandomNormal() {}
    };
   
}
#endif

