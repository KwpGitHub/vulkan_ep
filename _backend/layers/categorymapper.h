#ifndef CATEGORYMAPPER_H
#define CATEGORYMAPPER_H //CategoryMapper
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      cats_int64s, cats_strings, default_int64, default_string
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, int, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct CategoryMapper_parameter_descriptor{    
        Shape_t cats_int64s; Tensor* cats_strings; int default_int64; int default_string;
    };   

    struct CategoryMapper_input_desriptor{
        Tensor* X_input;
        
    };

    struct CategoryMapper_output_descriptor{
        Tensor* Y_output;
        
    };

    struct CategoryMapper_binding_descriptor{
        Shape_t cats_int64s; int default_int64; int default_string;
		Shape_t cats_strings;
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class CategoryMapper : public Layer {
        CategoryMapper_parameter_descriptor parameters;
        CategoryMapper_input_desriptor      input;
        CategoryMapper_output_descriptor    output;
        CategoryMapper_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, CategoryMapper_binding_descriptor>* program;
        
    public:
        CategoryMapper(std::string, CategoryMapper_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~CategoryMapper() {}

    };
}

//cpp stuff
namespace backend {    
   
    CategoryMapper::CategoryMapper(std::string n, CategoryMapper_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, CategoryMapper_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/categorymapper.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* CategoryMapper::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<CategoryMapper, Layer>(m, "CategoryMapper")
            .def("forward", &CategoryMapper::forward);    
    }*/
}

#endif
