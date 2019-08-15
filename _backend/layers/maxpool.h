#ifndef MAXPOOL_H
#define MAXPOOL_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
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

*/
//MaxPool
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         Indices_output_opt
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      auto_pad, ceil_mode, dilations, pads, storage_order, strides
//OPTIONAL_PARAMETERS_TYPE: int, int, Shape_t, Shape_t, int, Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class MaxPool : public Layer {
        typedef struct {    
            Shape_t kernel_shape; int auto_pad; int ceil_mode; Shape_t dilations; Shape_t pads; int storage_order; Shape_t strides;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            Tensor* Indices_output_opt;
        } output_descriptor;

        typedef struct {
            Shape_t kernel_shape; int auto_pad; int ceil_mode; Shape_t dilations; Shape_t pads; int storage_order; Shape_t strides;
		
            Shape_t X_input;
            
            Shape_t Y_output;
            Shape_t Indices_output_opt;
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MaxPool(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~MaxPool() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    MaxPool::MaxPool(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/maxpool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* MaxPool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void MaxPool::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
  		binding.Indices_output_opt = output.Indices_output_opt->shape();
 
		binding.kernel_shape = parameters.kernel_shape;
  		binding.auto_pad = parameters.auto_pad;
  		binding.ceil_mode = parameters.ceil_mode;
  		binding.dilations = parameters.dilations;
  		binding.pads = parameters.pads;
  		binding.storage_order = parameters.storage_order;
  		binding.strides = parameters.strides;
 
        program->bind(binding, *input.X_input->data(), *output.Y_output->data(), *output.Indices_output_opt->data());
    }
    
    void MaxPool::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<MaxPool, Layer>(m, "MaxPool")
            .def("forward", &MaxPool::forward);    
    }
}*/

#endif
