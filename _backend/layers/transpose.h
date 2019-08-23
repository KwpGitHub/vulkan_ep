#ifndef TRANSPOSE_H
#define TRANSPOSE_H 

#include "../layer.h"

/*

Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).

input: An input tensor.
output: Transposed output.
*/

//Transpose
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   transposed_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      perm
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>


//class stuff
namespace layers {   

    class Transpose : public backend::Layer {
        typedef struct {          
            backend::Shape_t data_i;
            
            backend::Shape_t transposed_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::vector<int> perm;
        std::string data_i;
        
        std::string transposed_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Transpose(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( std::vector<int> _perm); 
        virtual void bind(std::string _data_i, std::string _transposed_o); 
        virtual void build();

        ~Transpose() {}
    };
   
}
#endif

