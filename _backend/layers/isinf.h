#ifndef ISINF_H
#define ISINF_H 

#include "../layer.h"

/*
Map infinity to true and other values to false.
input: input
output: output
*/

//IsInf
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      detect_negative, detect_positive
//OPTIONAL_PARAMETERS_TYPE: int, int


//class stuff
namespace layers {   

    class IsInf : public backend::Layer {
        typedef struct {
            uint32_t size; float a;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        int detect_negative; int detect_positive;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
       

    public:
        IsInf(std::string name);
        
        virtual void forward();        
        virtual void init( int _detect_negative,  int _detect_positive); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~IsInf() {}
    };
   
}
#endif

