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
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string X_i; std::string Y_i;
        
        std::string Z_o;
        

        binding_descriptor   binding;
       

    public:
        Pow(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _X_i, std::string _Y_i, std::string _Z_o); 
        virtual void build();

        ~Pow() {}
    };
   
}
#endif

