#include "SVMClassifier.h"

//cpp stuff
namespace backend {    
   
    SVMClassifier::SVMClassifier(std::string n) : Layer(n) { }
       
    vuh::Device* SVMClassifier::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void SVMClassifier::init( Shape_t _classlabels_ints,  int _kernel_type,  int _post_transform,  Shape_t _vectors_per_class) {      
		 classlabels_ints = _classlabels_ints; 
 		 kernel_type = _kernel_type; 
 		 post_transform = _post_transform; 
 		 vectors_per_class = _vectors_per_class; 
  
    }
    
    void SVMClassifier::bind(std::string _classlabels_strings, std::string _coefficients, std::string _kernel_params, std::string _prob_a, std::string _prob_b, std::string _rho, std::string _support_vectors, std::string _X_input, std::string _Y_output, std::string _Z_output){
        classlabels_strings = _classlabels_strings; coefficients = _coefficients; kernel_params = _kernel_params; prob_a = _prob_a; prob_b = _prob_b; rho = _rho; support_vectors = _support_vectors; X_input = _X_input; Y_output = _Y_output; Z_output = _Z_output;
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
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/svmclassifier.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[classlabels_strings]->data(), *tensor_dict[coefficients]->data(), *tensor_dict[kernel_params]->data(), *tensor_dict[prob_a]->data(), *tensor_dict[prob_b]->data(), *tensor_dict[rho]->data(), *tensor_dict[support_vectors]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data(), *tensor_dict[Z_output]->data());
    }
    
}

    //backend::nn;

//python stuff


