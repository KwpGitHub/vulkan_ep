#ifndef SUB_H
#define SUB_H 

#include "../layer.h"

/*

Performs element-wise binary subtraction (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: First operand.
input: Second operand.
output: Result, has same element type as two inputs
*/

//Sub
//INPUTS:                   A_i, B_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   C_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Sub : public backend::Layer {
        typedef struct {          
            backend::Shape_t A_i; backend::Shape_t B_i;
            
            backend::Shape_t C_o;
            
        } binding_descriptor;
        using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;

        
        std::string A_i; std::string B_i;
        
        std::string C_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Sub(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _A_i, std::string _B_i, std::string _C_o); 
        virtual void build();

        ~Sub() {}
    };
   
}
#endif

