#ifndef CASTMAP_H
#define CASTMAP_H //CastMap
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      cast_to, map_form, max_map
//OPTIONAL_PARAMETERS_TYPE: int, int, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct CastMap_parameter_descriptor{    
        int cast_to; int map_form; int max_map;
    };   

    struct CastMap_input_desriptor{
        Tensor* X_input;
        
    };

    struct CastMap_output_descriptor{
        Tensor* Y_output;
        
    };

    struct CastMap_binding_descriptor{
        int cast_to; int map_form; int max_map;
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class CastMap : public Layer {
        CastMap_parameter_descriptor parameters;
        CastMap_input_desriptor      input;
        CastMap_output_descriptor    output;
        CastMap_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, CastMap_binding_descriptor>* program;
        
    public:
        CastMap(std::string, CastMap_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~CastMap() {}

    };
}

//cpp stuff
namespace backend {    
   
    CastMap::CastMap(std::string n, CastMap_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, CastMap_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/castmap.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* CastMap::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<CastMap, Layer>(m, "CastMap")
            .def("forward", &CastMap::forward);    
    }*/
}

#endif
