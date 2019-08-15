#ifndef GLOBALLPPOOL_H
#define GLOBALLPPOOL_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

 GlobalLpPool consumes an input tensor X and applies lp pool pooling across
 the values in the same channel. This is equivalent to LpPool with kernel size
 equal to the spatial dimension of input tensor.
input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
output: Output data tensor from pooling across the input tensor. Dimensions will be N x C x 1 x 1
//*/
//GlobalLpPool
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      p
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//class stuff
namespace backend {   

    class GlobalLpPool : public Layer {
        typedef struct {
            int p;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int p;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        GlobalLpPool(std::string n, int p);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~GlobalLpPool() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    GlobalLpPool::GlobalLpPool(std::string n, int p) : Layer(n) { }
       
    vuh::Device* GlobalLpPool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void GlobalLpPool::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.p = p;
 
    }
    
    void GlobalLpPool::call(std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/globallppool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<GlobalLpPool, Layer>(m, "GlobalLpPool")
            .def(py::init<std::string, int> ())
            .def("forward", &GlobalLpPool::forward)
            .def("init", &GlobalLpPool::init)
            .def("call", (void (GlobalLpPool::*) (std::string, std::string)) &GlobalLpPool::call);
    }
}

#endif

/* PYTHON STUFF

*/

