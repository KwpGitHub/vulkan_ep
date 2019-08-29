#ifndef NONZERO_H
#define NONZERO_H 

#include "../layer.h"

/*

    Returns the indices of the elements that are non-zero
    (in row-major order - by dimension).
    NonZero behaves similar to numpy.nonzero:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html

input: input
output: output (always 2D tensor)
*/

//NonZero
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class NonZero : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
       

    public:
        NonZero(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~NonZero() {}
    };
   
}
#endif

