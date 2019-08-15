#ifndef LPPOOL_H
#define LPPOOL_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

 LpPool consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.
input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
output: Output data tensor from Lp pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.
//*/
//LpPool
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      auto_pad, p, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, int, Shape_t, Shape_t

namespace py = pybind11;

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
        LpPool(std::string n, Shape_t kernel_shape, int auto_pad, int p, Shape_t pads, Shape_t strides);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~LpPool() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    LpPool::LpPool(std::string n, Shape_t kernel_shape, int auto_pad, int p, Shape_t pads, Shape_t strides) : Layer(n) { }
       
    vuh::Device* LpPool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void LpPool::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.kernel_shape = kernel_shape;
  		binding.auto_pad = auto_pad;
  		binding.p = p;
  		binding.pads = pads;
  		binding.strides = strides;
 
    }
    
    void LpPool::call(std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/lppool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<LpPool, Layer>(m, "LpPool")
            .def(py::init<std::string, Shape_t, int, int, Shape_t, Shape_t> ())
            .def("forward", &LpPool::forward)
            .def("init", &LpPool::init)
            .def("call", (void (LpPool::*) (std::string, std::string)) &LpPool::call);
    }
}

#endif

/* PYTHON STUFF

*/

