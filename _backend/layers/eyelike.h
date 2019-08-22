#ifndef EYELIKE_H
#define EYELIKE_H 

#include "../layer.h"

/*

Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
same as the input tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.

input: 2D input tensor to copy shape, and optionally, type information from.
output: Output tensor, same shape as input tensor T1.
*/

//EyeLike
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, k
//OPTIONAL_PARAMETERS_TYPE: int, int


//class stuff
namespace layers {   

    class EyeLike : public backend::Layer {
        typedef struct {          
            backend::Shape_t input_i;
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;

        int dtype; int k;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        EyeLike(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( int _dtype,  int _k); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~EyeLike() {}
    };
   
}
#endif

