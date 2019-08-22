#include "svmclassifier.h"
//cpp stuff
namespace layers {    
   
    SVMClassifier::SVMClassifier(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\svmclassifier.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* SVMClassifier::_get_device() {
        
        return backend::device;
    }
    
    void SVMClassifier::init( std::vector<int> _classlabels_ints,  std::vector<std::string> _classlabels_strings,  std::vector<float> _coefficients,  std::vector<float> _kernel_params,  std::string _kernel_type,  std::string _post_transform,  std::vector<float> _prob_a,  std::vector<float> _prob_b,  std::vector<float> _rho,  std::vector<float> _support_vectors,  std::vector<int> _vectors_per_class) {      
		 classlabels_ints = _classlabels_ints; 
 		 classlabels_strings = _classlabels_strings; 
 		 coefficients = _coefficients; 
 		 kernel_params = _kernel_params; 
 		 kernel_type = _kernel_type; 
 		 post_transform = _post_transform; 
 		 prob_a = _prob_a; 
 		 prob_b = _prob_b; 
 		 rho = _rho; 
 		 support_vectors = _support_vectors; 
 		 vectors_per_class = _vectors_per_class; 
  
    }
    
    void SVMClassifier::bind(std::string _X_i, std::string _Y_o, std::string _Z_o){
        X_i = _X_i; Y_o = _Y_o; Z_o = _Z_o;

		//binding.X_i = tensor_dict[X_i]->shape();
 
		//binding.Y_o = tensor_dict[Y_o]->shape();
  		//binding.Z_o = tensor_dict[Z_o]->shape();
 
		//binding.classlabels_ints = classlabels_ints;
  		//binding.classlabels_strings = classlabels_strings;
  		//binding.coefficients = coefficients;
  		//binding.kernel_params = kernel_params;
  		//binding.kernel_type = kernel_type;
  		//binding.post_transform = post_transform;
  		//binding.prob_a = prob_a;
  		//binding.prob_b = prob_b;
  		//binding.rho = rho;
  		//binding.support_vectors = support_vectors;
  		//binding.vectors_per_class = vectors_per_class;
         
    }

    void SVMClassifier::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data(), *tensor_dict[Z_o]->data());
    }

}

