#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Linear classifier

input: Data to be classified.
output: Classification outputs (one class per example).
output: Classification scores ([N,E] - one score for each class and example
//*/
//LinearClassifier
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output, Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               coefficients
//PARAMETER_TYPES:          Tensor*
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, intercepts, multi_class, post_transform
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*, int, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class LinearClassifier : public Layer {
        typedef struct {
            Shape_t classlabels_ints; int multi_class; int post_transform;
			Shape_t coefficients; Shape_t classlabels_strings; Shape_t intercepts;
            Shape_t X_input;
            
            Shape_t Y_output; Shape_t Z_output;
            
        } binding_descriptor;

        Shape_t classlabels_ints; int multi_class; int post_transform; std::string coefficients; std::string classlabels_strings; std::string intercepts;
        std::string X_input;
        
        std::string Y_output; std::string Z_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LinearClassifier(std::string n, Shape_t classlabels_ints, int multi_class, int post_transform);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string coefficients, std::string classlabels_strings, std::string intercepts, std::string X_input, std::string Y_output, std::string Z_output); 

        ~LinearClassifier() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    LinearClassifier::LinearClassifier(std::string n, Shape_t classlabels_ints, int multi_class, int post_transform) : Layer(n) { }
       
    vuh::Device* LinearClassifier::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void LinearClassifier::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
  		binding.Z_output = tensor_dict[Z_output]->shape();
 
		binding.classlabels_ints = classlabels_ints;
  		binding.multi_class = multi_class;
  		binding.post_transform = post_transform;
  		binding.coefficients = tensor_dict[coefficients]->shape();
  		binding.classlabels_strings = tensor_dict[classlabels_strings]->shape();
  		binding.intercepts = tensor_dict[intercepts]->shape();
 
    }
    
    void LinearClassifier::call(std::string coefficients, std::string classlabels_strings, std::string intercepts, std::string X_input, std::string Y_output, std::string Z_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/linearclassifier.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[coefficients]->data(), *tensor_dict[classlabels_strings]->data(), *tensor_dict[intercepts]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data(), *tensor_dict[Z_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<LinearClassifier, Layer>(m, "LinearClassifier")
            .def(py::init<std::string, Shape_t, int, int> ())
            .def("forward", &LinearClassifier::forward)
            .def("init", &LinearClassifier::init)
            .def("call", (void (LinearClassifier::*) (std::string, std::string, std::string, std::string, std::string, std::string)) &LinearClassifier::call);
    }
}

#endif

/* PYTHON STUFF

*/

