#include "SVMClassifier.h"

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

    py::module m("_backend.nn", "nn MOD");

//python stuff


