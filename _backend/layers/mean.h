#ifndef MEAN_H
#define MEAN_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for mean.
output: Output tensor.
*/

//Mean
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   mean_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Mean : public Layer {
        typedef struct {
            
			
            
            
            Shape_t mean_o;
            
        } binding_descriptor;

        
        
        
        std::string mean_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Mean(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _mean_o); 

        ~Mean() {}
    };

}

#endif

