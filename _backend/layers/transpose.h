#include "../layer.h"
#ifndef TRANSPOSE_H
#define TRANSPOSE_H 
/*

Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).

input: An input tensor.
output: Transposed output.
//*/
//Transpose
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   transposed_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      perm
//OPTIONAL_PARAMETERS_TYPE: Shape_t

//class stuff
namespace backend {   

    class Transpose : public Layer {
        typedef struct {
            Shape_t perm;
			
            Shape_t data_input;
            
            Shape_t transposed_output;
            
        } binding_descriptor;

        Shape_t perm;
        std::string data_input;
        
        std::string transposed_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Transpose(std::string n);
    
        void forward() { program->run(); }
        
        void init( Shape_t _perm); 
        void bind(std::string _data_input, std::string _transposed_output); 

        ~Transpose() {}

    };
    
}

#endif

