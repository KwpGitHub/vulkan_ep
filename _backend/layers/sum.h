#ifndef SUM_H
#define SUM_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for sum.
output: Output tensor.
*/

//Sum
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   sum_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Sum : public Layer {
        typedef struct {
            
			
            
            
            Shape_t sum_o;
            
        } binding_descriptor;

        
        
        
        std::string sum_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Sum(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _sum_o); 

        ~Sum() {}
    };

}

#endif

