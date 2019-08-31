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
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> m_perm;
        std::string m_data_i;
        
        std::string m_transposed_o;
        

        binding_descriptor   binding;
       

    public:
        Transpose(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _perm); 
        virtual void bind(std::string _data_i, std::string _transposed_o); 
        virtual void build();

        ~Transpose() {}
    };
   
}
#endif

