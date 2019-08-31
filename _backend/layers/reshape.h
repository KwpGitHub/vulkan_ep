#ifndef RESHAPE_H
#define RESHAPE_H 

#include "../layer.h"

/*

Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor).
input: An input tensor.
input: Specified shape for output.
output: Reshaped data.
*/

//Reshape
//INPUTS:                   data_i, shape_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   reshaped_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Reshape : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string m_data_i; std::string m_shape_i;
        
        std::string m_reshaped_o;
        

        binding_descriptor   binding;
       

    public:
        Reshape(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _data_i, std::string _shape_i, std::string _reshaped_o); 
        virtual void build();

        ~Reshape() {}
    };
   
}
#endif

