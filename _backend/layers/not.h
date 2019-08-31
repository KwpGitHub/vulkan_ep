#ifndef NOT_H
#define NOT_H 

#include "../layer.h"

/*

Returns the negation of the input tensor element-wise.

input: Input tensor
output: Output tensor
*/

//Not
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

    class Not : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        Not(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~Not() {}
    };
   
}
#endif

