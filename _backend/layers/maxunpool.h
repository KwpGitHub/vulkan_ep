#ifndef MAXUNPOOL_H
#define MAXUNPOOL_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corrsponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corrsponding
 pooling op that the unpooling op is trying to invert.

input: Input data tensor that has to be unpooled. This tensor is typically the first output of the MaxPool op.Dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non-image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
input: Input data tensor containing the indices corresponding to elements in the first input tensor X.This tensor is typically the second output of the MaxPool op.Dimensions must be the same as input tensor X. The indices are linear, i.e. computed considering the tensor as flattened 1-D tensor, assuming row-major storage. Also, the linear indices should not consider padding. So the values in indices are in the range [0, N x C x D1 x ... x Dn).
input: The shape of the output can be explicitly set which will cause pads values to be auto generated. If 'output_shape' is specified, 'pads' values are ignored.
output: Output data tensor that contains the result of the unpooling.
//*/
//MaxUnpool
//INPUTS:                   X_input, I_input
//OPTIONAL_INPUTS:          output_shape_input_opt
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      pads, strides
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class MaxUnpool : public Layer {
        typedef struct {
            Shape_t kernel_shape; Shape_t pads; Shape_t strides;
			
            Shape_t X_input; Shape_t I_input;
            Shape_t output_shape_input_opt;
            Shape_t output_output;
            
        } binding_descriptor;

        Shape_t kernel_shape; Shape_t pads; Shape_t strides;
        std::string X_input; std::string I_input;
        std::string output_shape_input_opt;
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MaxUnpool(std::string n, Shape_t kernel_shape, Shape_t pads, Shape_t strides);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string I_input, std::string output_shape_input_opt, std::string output_output); 

        ~MaxUnpool() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    MaxUnpool::MaxUnpool(std::string n, Shape_t kernel_shape, Shape_t pads, Shape_t strides) : Layer(n) { }
       
    vuh::Device* MaxUnpool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void MaxUnpool::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.I_input = tensor_dict[I_input]->shape();
  		binding.output_shape_input_opt = tensor_dict[output_shape_input_opt]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.kernel_shape = kernel_shape;
  		binding.pads = pads;
  		binding.strides = strides;
 
    }
    
    void MaxUnpool::call(std::string X_input, std::string I_input, std::string output_shape_input_opt, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/maxunpool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[I_input]->data(), *tensor_dict[output_shape_input_opt]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<MaxUnpool, Layer>(m, "MaxUnpool")
            .def(py::init<std::string, Shape_t, Shape_t, Shape_t> ())
            .def("forward", &MaxUnpool::forward)
            .def("init", &MaxUnpool::init)
            .def("call", (void (MaxUnpool::*) (std::string, std::string, std::string, std::string)) &MaxUnpool::call);
    }
}

#endif

/* PYTHON STUFF

*/

