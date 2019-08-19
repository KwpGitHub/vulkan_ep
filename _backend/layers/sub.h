#ifndef SUB_H
#define SUB_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Performs element-wise binary subtraction (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: First operand.
input: Second operand.
output: Result, has same element type as two inputs
*/

//Sub
//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   C_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Sub : public Layer {
        typedef struct {
            
			
            Shape_t A_input; Shape_t B_input;
            
            Shape_t C_output;
            
        } binding_descriptor;

        
        std::string A_input; std::string B_input;
        
        std::string C_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Sub();
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _A_input, std::string _B_input, std::string _C_output); 

        ~Sub() {}
    };

    
    void init_layer_Sub(py::module& m) {
        // py::class_(m, "Sub");
    }
    

}


#endif

