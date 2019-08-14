#ifndef FLOOR_H
#define FLOOR_H //Floor
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Floor_parameter_descriptor{    
        
    };   

    struct Floor_input_desriptor{
        Tensor* X_input;
        
    };

    struct Floor_output_descriptor{
        Tensor* Y_output;
        
    };

    struct Floor_binding_descriptor{
        
		
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class Floor : public Layer {
        Floor_parameter_descriptor parameters;
        Floor_input_desriptor      input;
        Floor_output_descriptor    output;
        Floor_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Floor_binding_descriptor>* program;
        
    public:
        Floor(std::string, Floor_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Floor() {}

    };
}

//cpp stuff
namespace backend {    
   
    Floor::Floor(std::string n, Floor_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Floor_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/floor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Floor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Floor, Layer>(m, "Floor")
            .def("forward", &Floor::forward);    
    }*/
}

#endif
