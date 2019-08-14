#ifndef SCAN_H
#define SCAN_H //Scan
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               body, num_scan_inputs
//PARAMETER_TYPES:          int, int
//OPTIONAL_PARAMETERS:      scan_input_axes, scan_input_directions, scan_output_axes, scan_output_directions
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Shape_t, Shape_t, Shape_t

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Scan_parameter_descriptor{    
        int body; int num_scan_inputs; Shape_t scan_input_axes; Shape_t scan_input_directions; Shape_t scan_output_axes; Shape_t scan_output_directions;
    };   

    struct Scan_input_desriptor{
        
        
    };

    struct Scan_output_descriptor{
        
        
    };

    struct Scan_binding_descriptor{
        int body; int num_scan_inputs; Shape_t scan_input_axes; Shape_t scan_input_directions; Shape_t scan_output_axes; Shape_t scan_output_directions;
		
        
        
        
        
    };
}


namespace backend {

    class Scan : public Layer {
        Scan_parameter_descriptor parameters;
        Scan_input_desriptor      input;
        Scan_output_descriptor    output;
        Scan_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Scan_binding_descriptor>* program;
        
    public:
        Scan(std::string, Scan_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Scan() {}

    };
}

//cpp stuff
namespace backend {    
   
    Scan::Scan(std::string n, Scan_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Scan_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/scan.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Scan::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Scan, Layer>(m, "Scan")
            .def("forward", &Scan::forward);    
    }*/
}

#endif
