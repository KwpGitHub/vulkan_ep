#ifndef POW_H
#define POW_H 

#include "../layer.h"

/*

Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
input: First operand, base of the exponent.
input: Second operand, power of the exponent.
output: Output tensor (same size as X)
*/

//Pow
//INPUTS:                   X_i, Y_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Pow : public backend::Layer {
        typedef struct {          
            backend::Shape_t X_i; backend::Shape_t Y_i;
            
            backend::Shape_t Z_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        
        std::string X_i; std::string Y_i;
        
        std::string Z_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Pow(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _X_i, std::string _Y_i, std::string _Z_o); 
        virtual void build();

        ~Pow() {}
    };
   
}
#endif

