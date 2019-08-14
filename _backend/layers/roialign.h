#ifndef ROIALIGN_H
#define ROIALIGN_H //RoiAlign
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input, rois_input, batch_indices_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      mode, output_height, output_width, sampling_ratio, spatial_scale
//OPTIONAL_PARAMETERS_TYPE: int, int, int, int, float

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct RoiAlign_parameter_descriptor{    
        int mode; int output_height; int output_width; int sampling_ratio; float spatial_scale;
    };   

    struct RoiAlign_input_desriptor{
        Tensor* X_input; Tensor* rois_input; Tensor* batch_indices_input;
        
    };

    struct RoiAlign_output_descriptor{
        Tensor* Y_output;
        
    };

    struct RoiAlign_binding_descriptor{
        int mode; int output_height; int output_width; int sampling_ratio; float spatial_scale;
		
        Shape_t X_input; Shape_t rois_input; Shape_t batch_indices_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class RoiAlign : public Layer {
        RoiAlign_parameter_descriptor parameters;
        RoiAlign_input_desriptor      input;
        RoiAlign_output_descriptor    output;
        RoiAlign_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, RoiAlign_binding_descriptor>* program;
        
    public:
        RoiAlign(std::string, RoiAlign_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~RoiAlign() {}

    };
}

//cpp stuff
namespace backend {    
   
    RoiAlign::RoiAlign(std::string n, RoiAlign_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, RoiAlign_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/roialign.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* RoiAlign::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<RoiAlign, Layer>(m, "RoiAlign")
            .def("forward", &RoiAlign::forward);    
    }*/
}

#endif
