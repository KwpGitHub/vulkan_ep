#ifndef ZIPMAP_H
#define ZIPMAP_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Creates a map from the input and the attributes.<br>
    The values are provided by the input tensor, while the keys are specified by the attributes.
    Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
    The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>

input: The input values
output: The output map
//*/
//ZipMap
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_int64s, classlabels_strings
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*

namespace py = pybind11;

//class stuff
namespace backend {   

    class ZipMap : public Layer {
        typedef struct {
            Shape_t classlabels_int64s;
			Shape_t classlabels_strings;
            Shape_t X_input;
            
            Shape_t Z_output;
            
        } binding_descriptor;

        Shape_t classlabels_int64s; std::string classlabels_strings;
        std::string X_input;
        
        std::string Z_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ZipMap(std::string n, Shape_t classlabels_int64s);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string classlabels_strings, std::string X_input, std::string Z_output); 

        ~ZipMap() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ZipMap::ZipMap(std::string n, Shape_t classlabels_int64s) : Layer(n) { }
       
    vuh::Device* ZipMap::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ZipMap::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Z_output = tensor_dict[Z_output]->shape();
 
		binding.classlabels_int64s = classlabels_int64s;
  		binding.classlabels_strings = tensor_dict[classlabels_strings]->shape();
 
    }
    
    void ZipMap::call(std::string classlabels_strings, std::string X_input, std::string Z_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/zipmap.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[classlabels_strings]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Z_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ZipMap, Layer>(m, "ZipMap")
            .def(py::init<std::string, Shape_t> ())
            .def("forward", &ZipMap::forward)
            .def("init", &ZipMap::init)
            .def("call", (void (ZipMap::*) (std::string, std::string, std::string)) &ZipMap::call);
    }
}

#endif

/* PYTHON STUFF

*/

