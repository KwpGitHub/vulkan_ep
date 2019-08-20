#ifndef AVERAGEPOOL_H
#define AVERAGEPOOL_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

 AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
 
input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
output: Output data tensor from average or max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used
*/

//AveragePool
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      auto_pad, ceil_mode, count_include_pad, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, int, int, Shape_t, Shape_t

//class stuff
namespace backend {   

    class AveragePool : public Layer {
        typedef struct {
            Shape_t kernel_shape; int auto_pad; int ceil_mode; int count_include_pad; Shape_t pads; Shape_t strides;
			
            Shape_t X_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        Shape_t kernel_shape; int auto_pad; int ceil_mode; int count_include_pad; Shape_t pads; Shape_t strides;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        AveragePool(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( Shape_t _kernel_shape,  int _auto_pad,  int _ceil_mode,  int _count_include_pad,  Shape_t _pads,  Shape_t _strides); 
        void bind(std::string _X_i, std::string _Y_o); 

        ~AveragePool() {}
    };

}

#endif

