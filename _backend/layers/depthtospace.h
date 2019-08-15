#ifndef DEPTHTOSPACE_H
#define DEPTHTOSPACE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*
DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions.

input: Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.
output: Output tensor of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].
//*/
//DepthToSpace
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               blocksize
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class DepthToSpace : public Layer {
        typedef struct {
            int blocksize;
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int blocksize;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        DepthToSpace(std::string n, int blocksize);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string output_output); 

        ~DepthToSpace() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    DepthToSpace::DepthToSpace(std::string n, int blocksize) : Layer(n) { }
       
    vuh::Device* DepthToSpace::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void DepthToSpace::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.blocksize = blocksize;
 
    }
    
    void DepthToSpace::call(std::string input_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/depthtospace.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<DepthToSpace, Layer>(m, "DepthToSpace")
            .def(py::init<std::string, int> ())
            .def("forward", &DepthToSpace::forward)
            .def("init", &DepthToSpace::init)
            .def("call", (void (DepthToSpace::*) (std::string, std::string)) &DepthToSpace::call);
    }
}

#endif

/* PYTHON STUFF

*/

