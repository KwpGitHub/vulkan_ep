#include "../layer.h"
#ifndef EYELIKE_H
#define EYELIKE_H 
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
//*/
//EyeLike
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, k
//OPTIONAL_PARAMETERS_TYPE: int, int

//class stuff
namespace backend {   

    class EyeLike : public Layer {
        typedef struct {
            int dtype; int k;
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int dtype; int k;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        EyeLike(std::string n);
    
        void forward() { program->run(); }
        
        void init( int _dtype,  int _k); 
        void bind(std::string _input_input, std::string _output_output); 

        ~EyeLike() {}

    };
    
}

#endif

