#ifndef NONMAXSUPPRESSION_H
#define NONMAXSUPPRESSION_H //NonMaxSuppression
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   boxes_input, scores_input
//OPTIONAL_INPUTS:          max_output_boxes_per_class_input_opt, iou_threshold_input_opt, score_threshold_input_opt
//OUTPUS:                   selected_indices_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      center_point_box
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct NonMaxSuppression_parameter_descriptor{    
        int center_point_box;
    };   

    struct NonMaxSuppression_input_desriptor{
        Tensor* boxes_input; Tensor* scores_input;
        Tensor* max_output_boxes_per_class_input_opt; Tensor* iou_threshold_input_opt; Tensor* score_threshold_input_opt;
    };

    struct NonMaxSuppression_output_descriptor{
        Tensor* selected_indices_output;
        
    };

    struct NonMaxSuppression_binding_descriptor{
        int center_point_box;
		
        Shape_t boxes_input; Shape_t scores_input;
        Shape_t max_output_boxes_per_class_input_opt; Shape_t iou_threshold_input_opt; Shape_t score_threshold_input_opt;
        Shape_t selected_indices_output;
        
    };
}


namespace backend {

    class NonMaxSuppression : public Layer {
        NonMaxSuppression_parameter_descriptor parameters;
        NonMaxSuppression_input_desriptor      input;
        NonMaxSuppression_output_descriptor    output;
        NonMaxSuppression_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, NonMaxSuppression_binding_descriptor>* program;
        
    public:
        NonMaxSuppression(std::string, NonMaxSuppression_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~NonMaxSuppression() {}

    };
}

//cpp stuff
namespace backend {    
   
    NonMaxSuppression::NonMaxSuppression(std::string n, NonMaxSuppression_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, NonMaxSuppression_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/nonmaxsuppression.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* NonMaxSuppression::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<NonMaxSuppression, Layer>(m, "NonMaxSuppression")
            .def("forward", &NonMaxSuppression::forward);    
    }*/
}

#endif
