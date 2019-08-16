#include "../layer.h"
#ifndef MAXPOOL_H
#define MAXPOOL_H 
/*

 MaxPool consumes an input tensor X and applies max pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 max pooling consisting of computing the max on all values of a
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
 The output of each pooling window is maximum number of elements exclude pad.
 
input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
output: Output data tensor from average or max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used
output: Indices tensor from max pooling across the input tensor. The dimensions of indices are the same as output tensor. The values in indices of are the indices of the selected values during pooling. The indices are computed as flatten 1-D tensor, and the indices do not consider padding. So the values in indices are in [0, N x C x D1 x ... x Dn).
//*/
//MaxPool
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         Indices_output_opt
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      auto_pad, ceil_mode, dilations, pads, storage_order, strides
//OPTIONAL_PARAMETERS_TYPE: int, int, Shape_t, Shape_t, int, Shape_t

//class stuff
namespace backend {   

    class MaxPool : public Layer {
        typedef struct {
            Shape_t kernel_shape; int auto_pad; int ceil_mode; Shape_t dilations; Shape_t pads; int storage_order; Shape_t strides;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            Shape_t Indices_output_opt;
        } binding_descriptor;

        Shape_t kernel_shape; int auto_pad; int ceil_mode; Shape_t dilations; Shape_t pads; int storage_order; Shape_t strides;
        std::string X_input;
        
        std::string Y_output;
        std::string Indices_output_opt;

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MaxPool(std::string n, Shape_t kernel_shape, int auto_pad, int ceil_mode, Shape_t dilations, Shape_t pads, int storage_order, Shape_t strides);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output, std::string Indices_output_opt); 

        ~MaxPool() {}

    };
    
}

#endif

