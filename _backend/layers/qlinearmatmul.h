#ifndef QLINEARMATMUL_H
#define QLINEARMATMUL_H //QLinearMatMul

#include "../layer.h"

//INPUTS:                   a_input, a_scale_input, a_zero_point_input, b_input, b_scale_input, b_zero_point_input, y_scale_input, y_zero_point_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class QLinearMatMul : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			
            //input
            Shape_t a_input; Shape_t a_scale_input; Shape_t a_zero_point_input; Shape_t b_input; Shape_t b_scale_input; Shape_t b_zero_point_input; Shape_t y_scale_input; Shape_t y_zero_point_input;
            
            //output
            Shape_t y_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        QLinearMatMul(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        
		
        //input
        std::string a_input; std::string a_scale_input; std::string a_zero_point_input; std::string b_input; std::string b_scale_input; std::string b_zero_point_input; std::string y_scale_input; std::string y_zero_point_input;
        
        //output
        std::string y_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~QLinearMatMul() {}
    };
}


namespace backend {    
    QLinearMatMul::QLinearMatMul(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/qlinearmatmul.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({tensor_dict[a_input]->shape(), tensor_dict[a_scale_input]->shape(), tensor_dict[a_zero_point_input]->shape(), tensor_dict[b_input]->shape(), tensor_dict[b_scale_input]->shape(), tensor_dict[b_zero_point_input]->shape(), tensor_dict[y_scale_input]->shape(), tensor_dict[y_zero_point_input]->shape(), tensor_dict[y_output]->shape()} 
                        
                        , tensor_dict[a_input], tensor_dict[a_scale_input], tensor_dict[a_zero_point_input], tensor_dict[b_input], tensor_dict[b_scale_input], tensor_dict[b_zero_point_input], tensor_dict[y_scale_input], tensor_dict[y_zero_point_input], tensor_dict[y_output] );
    }

    vuh::Device* QLinearMatMul::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
