#ifndef ZIPMAP_H
#define ZIPMAP_H //ZipMap
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_int64s, classlabels_strings
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct ZipMap_parameter_descriptor{    
        Shape_t classlabels_int64s; Tensor* classlabels_strings;
    };   

    struct ZipMap_input_desriptor{
        Tensor* X_input;
        
    };

    struct ZipMap_output_descriptor{
        Tensor* Z_output;
        
    };

    struct ZipMap_binding_descriptor{
        Shape_t classlabels_int64s;
		Shape_t classlabels_strings;
        Shape_t X_input;
        
        Shape_t Z_output;
        
    };
}


namespace backend {

    class ZipMap : public Layer {
        ZipMap_parameter_descriptor parameters;
        ZipMap_input_desriptor      input;
        ZipMap_output_descriptor    output;
        ZipMap_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, ZipMap_binding_descriptor>* program;
        
    public:
        ZipMap(std::string, ZipMap_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~ZipMap() {}

    };
}

//cpp stuff
namespace backend {    
   
    ZipMap::ZipMap(std::string n, ZipMap_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, ZipMap_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/zipmap.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* ZipMap::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<ZipMap, Layer>(m, "ZipMap")
            .def("forward", &ZipMap::forward);    
    }*/
}

#endif
