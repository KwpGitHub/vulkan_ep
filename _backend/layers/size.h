#ifndef SIZE_H
#define SIZE_H //Size
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   size_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct Size_parameter_descriptor{    
        
    };   

    struct Size_input_desriptor{
        Tensor* data_input;
        
    };

    struct Size_output_descriptor{
        Tensor* size_output;
        
    };

    struct Size_binding_descriptor{
        
		
        Shape_t data_input;
        
        Shape_t size_output;
        
    };
}


namespace backend {

    class Size : public Layer {
        Size_parameter_descriptor parameters;
        Size_input_desriptor      input;
        Size_output_descriptor    output;
        Size_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, Size_binding_descriptor>* program;
        
    public:
        Size(std::string, Size_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~Size() {}

    };
}

//cpp stuff
namespace backend {    
   
    Size::Size(std::string n, Size_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, Size_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/size.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* Size::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<Size, Layer>(m, "Size")
            .def("forward", &Size::forward);    
    }*/
}

#endif
