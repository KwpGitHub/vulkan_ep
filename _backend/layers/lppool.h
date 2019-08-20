#ifndef LPPOOL_H
#define LPPOOL_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

 LpPool consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.
input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
output: Output data tensor from Lp pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.
*/

//LpPool
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      auto_pad, p, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, int, Shape_t, Shape_t

//class stuff
namespace backend {   

    class LpPool : public Layer {
        typedef struct {
            Shape_t kernel_shape; int auto_pad; int p; Shape_t pads; Shape_t strides;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        Shape_t kernel_shape; int auto_pad; int p; Shape_t pads; Shape_t strides;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LpPool(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( Shape_t _kernel_shape,  int _auto_pad,  int _p,  Shape_t _pads,  Shape_t _strides); 
        void bind(std::string _X_input, std::string _Y_output); 

        ~LpPool() {}
    };

}

#endif

