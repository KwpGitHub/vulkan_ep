#ifndef LINEARREGRESSOR_H
#define LINEARREGRESSOR_H //LinearRegressor

#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      coefficients, intercepts, post_transform, targets
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, int, int



namespace backend {
    class LinearRegressor : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int post_transform; int targets;
			Shape_t coefficients; Shape_t intercepts;
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        LinearRegressor(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        Tensor* coefficients; Tensor* intercepts; int post_transform; int targets;
		Shape_t coefficients_s; Shape_t intercepts_s;
        //input
        std::string X_input;
        
        //output
        std::string Y_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~LinearRegressor() {}
    };
}


namespace backend {    
    LinearRegressor::LinearRegressor(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/linearregressor.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({post_transform, targets, coefficients_s, intercepts_s, tensor_dict[X_input]->shape(), tensor_dict[Y_output]->shape()} 
                        , *coefficients, *intercepts
                        , tensor_dict[X_input], tensor_dict[Y_output] );
    }

    vuh::Device* LinearRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
