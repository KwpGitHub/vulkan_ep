#ifndef CONSTANTOFSHAPE_H
#define CONSTANTOFSHAPE_H //ConstantOfShape
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      value
//OPTIONAL_PARAMETERS_TYPE: Tensor*

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct ConstantOfShape_parameter_descriptor{    
        Tensor* value;
    };   

    struct ConstantOfShape_input_desriptor{
        Tensor* input_input;
        
    };

    struct ConstantOfShape_output_descriptor{
        Tensor* output_output;
        
    };

    struct ConstantOfShape_binding_descriptor{
        
		Shape_t value;
        Shape_t input_input;
        
        Shape_t output_output;
        
    };
}


namespace backend {

    class ConstantOfShape : public Layer {
        ConstantOfShape_parameter_descriptor parameters;
        ConstantOfShape_input_desriptor      input;
        ConstantOfShape_output_descriptor    output;
        ConstantOfShape_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, ConstantOfShape_binding_descriptor>* program;
        
    public:
        ConstantOfShape(std::string, ConstantOfShape_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~ConstantOfShape() {}

    };
}

//cpp stuff
namespace backend {    
   
    ConstantOfShape::ConstantOfShape(std::string n, ConstantOfShape_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, ConstantOfShape_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/constantofshape.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* ConstantOfShape::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<ConstantOfShape, Layer>(m, "ConstantOfShape")
            .def("forward", &ConstantOfShape::forward);    
    }*/
}

#endif
