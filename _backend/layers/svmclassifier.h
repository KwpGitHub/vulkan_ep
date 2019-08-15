#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Support Vector Machine classifier

input: Data to be classified.
output: Classification outputs (one class per example).
output: Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for each class, otherwise they are raw scores.
//*/
//SVMClassifier
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output, Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, coefficients, kernel_params, kernel_type, post_transform, prob_a, prob_b, rho, support_vectors, vectors_per_class
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*, Tensor*, int, int, Tensor*, Tensor*, Tensor*, Tensor*, Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class SVMClassifier : public Layer {
        typedef struct {
            Shape_t classlabels_ints; int kernel_type; int post_transform; Shape_t vectors_per_class;
			Shape_t classlabels_strings; Shape_t coefficients; Shape_t kernel_params; Shape_t prob_a; Shape_t prob_b; Shape_t rho; Shape_t support_vectors;
            Shape_t X_input;
            
            Shape_t Y_output; Shape_t Z_output;
            
        } binding_descriptor;

        Shape_t classlabels_ints; int kernel_type; int post_transform; Shape_t vectors_per_class; std::string classlabels_strings; std::string coefficients; std::string kernel_params; std::string prob_a; std::string prob_b; std::string rho; std::string support_vectors;
        std::string X_input;
        
        std::string Y_output; std::string Z_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        SVMClassifier(std::string n, Shape_t classlabels_ints, int kernel_type, int post_transform, Shape_t vectors_per_class);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string classlabels_strings, std::string coefficients, std::string kernel_params, std::string prob_a, std::string prob_b, std::string rho, std::string support_vectors, std::string X_input, std::string Y_output, std::string Z_output); 

        ~SVMClassifier() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    SVMClassifier::SVMClassifier(std::string n, Shape_t classlabels_ints, int kernel_type, int post_transform, Shape_t vectors_per_class) : Layer(n) { }
       
    vuh::Device* SVMClassifier::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void SVMClassifier::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
  		binding.Z_output = tensor_dict[Z_output]->shape();
 
		binding.classlabels_ints = classlabels_ints;
  		binding.kernel_type = kernel_type;
  		binding.post_transform = post_transform;
  		binding.vectors_per_class = vectors_per_class;
  		binding.classlabels_strings = tensor_dict[classlabels_strings]->shape();
  		binding.coefficients = tensor_dict[coefficients]->shape();
  		binding.kernel_params = tensor_dict[kernel_params]->shape();
  		binding.prob_a = tensor_dict[prob_a]->shape();
  		binding.prob_b = tensor_dict[prob_b]->shape();
  		binding.rho = tensor_dict[rho]->shape();
  		binding.support_vectors = tensor_dict[support_vectors]->shape();
 
    }
    
    void SVMClassifier::call(std::string classlabels_strings, std::string coefficients, std::string kernel_params, std::string prob_a, std::string prob_b, std::string rho, std::string support_vectors, std::string X_input, std::string Y_output, std::string Z_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/svmclassifier.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[classlabels_strings]->data(), *tensor_dict[coefficients]->data(), *tensor_dict[kernel_params]->data(), *tensor_dict[prob_a]->data(), *tensor_dict[prob_b]->data(), *tensor_dict[rho]->data(), *tensor_dict[support_vectors]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data(), *tensor_dict[Z_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<SVMClassifier, Layer>(m, "SVMClassifier")
            .def(py::init<std::string, Shape_t, int, int, Shape_t> ())
            .def("forward", &SVMClassifier::forward)
            .def("init", &SVMClassifier::init)
            .def("call", (void (SVMClassifier::*) (std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string)) &SVMClassifier::call);
    }
}

#endif

/* PYTHON STUFF

*/

