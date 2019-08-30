#ifndef LPNORMALIZATION_H
#define LPNORMALIZATION_H 

#include "../layer.h"

/*

Given a matrix, apply Lp-normalization along the provided axis.

input: Input matrix
output: Matrix after normalization
*/

//LpNormalization
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, p
//OPTIONAL_PARAMETERS_TYPE: int, int


//class stuff
namespace layers {   

    class LpNormalization : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        int m_axis; int m_p;
        std::string m_input_i;
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        LpNormalization(std::string name);
        
        virtual void forward();        
        virtual void init( int _axis,  int _p); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~LpNormalization() {}
    };
   
}
#endif

