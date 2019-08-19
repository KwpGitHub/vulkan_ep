#ifndef GLOBALAVERAGEPOOL_H
#define GLOBALAVERAGEPOOL_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

 GlobalAveragePool consumes an input tensor X and applies average pooling across
 the values in the same channel. This is equivalent to AveragePool with kernel size
 equal to the spatial dimension of input tensor.
input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
output: Output data tensor from pooling across the input tensor. Dimensions will be N x C x 1 x 1
*/

//GlobalAveragePool
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

    class GlobalAveragePool : public Layer {
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
        GlobalAveragePool();
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _X_input, std::string _Y_output); 

        ~GlobalAveragePool() {}
    };

    
    void init_layer_GlobalAveragePool(py::module& m) {
        // py::class_(m, "GlobalAveragePool");
    }
    

}


#endif

