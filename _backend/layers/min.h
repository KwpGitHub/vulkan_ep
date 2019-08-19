#ifndef MIN_H
#define MIN_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for min.
output: Output tensor.
*/

//Min
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   min_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Min : public Layer {
        typedef struct {
            
			
            
            
            Shape_t min_output;
            
        } binding_descriptor;

        
        
        
        std::string min_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Min();
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _min_output); 

        ~Min() {}
    };

    
    void init_layer_Min(py::module& m) {
        // py::class_(m, "Min");
    }
    

}


#endif

