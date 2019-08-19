#ifndef RELU_H
#define RELU_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.

input: Input tensor
output: Output tensor
*/

//Relu
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Relu : public Layer {
        typedef struct {
            
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Relu();
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _X_input, std::string _Y_output); 

        ~Relu() {}
    };

    
    void init_layer_Relu(py::module& m) {
        // py::class_(m, "Relu");
    }
    

}


#endif

