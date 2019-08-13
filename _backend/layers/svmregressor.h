#ifndef SVMREGRESSOR_H
#define SVMREGRESSOR_H //SVMRegressor

#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      coefficients, kernel_params, kernel_type, n_supports, one_class, post_transform, rho, support_vectors
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, int, int, int, int, Tensor*, Tensor*



namespace backend {
    class SVMRegressor : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int kernel_type; int n_supports; int one_class; int post_transform;
			Shape_t coefficients; Shape_t kernel_params; Shape_t rho; Shape_t support_vectors;
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        SVMRegressor(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        Tensor* coefficients; Tensor* kernel_params; int kernel_type; int n_supports; int one_class; int post_transform; Tensor* rho; Tensor* support_vectors;
		Shape_t coefficients_s; Shape_t kernel_params_s; Shape_t rho_s; Shape_t support_vectors_s;
        //input
        std::string X_input;
        
        //output
        std::string Y_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~SVMRegressor() {}
    };
}


namespace backend {    
    SVMRegressor::SVMRegressor(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/svmregressor.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({kernel_type, n_supports, one_class, post_transform, coefficients_s, kernel_params_s, rho_s, support_vectors_s, tensor_dict[X_input]->shape(), tensor_dict[Y_output]->shape()} 
                        , *coefficients, *kernel_params, *rho, *support_vectors
                        , tensor_dict[X_input], tensor_dict[Y_output] );
    }

    vuh::Device* SVMRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
