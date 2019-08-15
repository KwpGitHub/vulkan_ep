#ifndef MAXROIPOOL_H
#define MAXROIPOOL_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

 ROI max pool consumes an input tensor X and region of interests (RoIs) to
 apply max pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).
input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data.
input: RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of shape (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...].
output: RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0], pooled_shape[1]).
//*/
//MaxRoiPool
//INPUTS:                   X_input, rois_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               pooled_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      spatial_scale
//OPTIONAL_PARAMETERS_TYPE: float

namespace py = pybind11;

//class stuff
namespace backend {   

    class MaxRoiPool : public Layer {
        typedef struct {
            Shape_t pooled_shape; float spatial_scale;
			
            Shape_t X_input; Shape_t rois_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        Shape_t pooled_shape; float spatial_scale;
        std::string X_input; std::string rois_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MaxRoiPool(std::string n, Shape_t pooled_shape, float spatial_scale);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string rois_input, std::string Y_output); 

        ~MaxRoiPool() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    MaxRoiPool::MaxRoiPool(std::string n, Shape_t pooled_shape, float spatial_scale) : Layer(n) { }
       
    vuh::Device* MaxRoiPool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void MaxRoiPool::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.rois_input = tensor_dict[rois_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.pooled_shape = pooled_shape;
  		binding.spatial_scale = spatial_scale;
 
    }
    
    void MaxRoiPool::call(std::string X_input, std::string rois_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/maxroipool.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[rois_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<MaxRoiPool, Layer>(m, "MaxRoiPool")
            .def(py::init<std::string, Shape_t, float> ())
            .def("forward", &MaxRoiPool::forward)
            .def("init", &MaxRoiPool::init)
            .def("call", (void (MaxRoiPool::*) (std::string, std::string, std::string)) &MaxRoiPool::call);
    }
}

#endif

/* PYTHON STUFF

*/

