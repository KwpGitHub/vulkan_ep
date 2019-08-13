#ifndef QLINEARCONV_H
#define QLINEARCONV_H //QLinearConv

#include "../layer.h"

//INPUTS:                   x_input, x_scale_input, x_zero_point_input, w_input, w_scale_input, w_zero_point_input, y_scale_input, y_zero_point_input
//OPTIONAL_INPUTS:          B_input_o
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      auto_pad, dilations, group, kernel_shape, pads, strides
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t, int, Shape_t, Shape_t, Shape_t



namespace backend {
    class QLinearConv : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
			
            //input
            Shape_t x_input; Shape_t x_scale_input; Shape_t x_zero_point_input; Shape_t w_input; Shape_t w_scale_input; Shape_t w_zero_point_input; Shape_t y_scale_input; Shape_t y_zero_point_input;
            Shape_t B_input_o;
            //output
            Shape_t y_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        QLinearConv(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
		
        //input
        std::string x_input; std::string x_scale_input; std::string x_zero_point_input; std::string w_input; std::string w_scale_input; std::string w_zero_point_input; std::string y_scale_input; std::string y_zero_point_input;
        std::string B_input_o;
        //output
        std::string y_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~QLinearConv() {}
    };
}


namespace backend {    
    QLinearConv::QLinearConv(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/qlinearconv.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({auto_pad, dilations, group, kernel_shape, pads, strides, tensor_dict[x_input]->shape(), tensor_dict[x_scale_input]->shape(), tensor_dict[x_zero_point_input]->shape(), tensor_dict[w_input]->shape(), tensor_dict[w_scale_input]->shape(), tensor_dict[w_zero_point_input]->shape(), tensor_dict[y_scale_input]->shape(), tensor_dict[y_zero_point_input]->shape(), tensor_dict[B_input_o]->shape(), tensor_dict[y_output]->shape()} 
                        
                        , tensor_dict[x_input], tensor_dict[x_scale_input], tensor_dict[x_zero_point_input], tensor_dict[w_input], tensor_dict[w_scale_input], tensor_dict[w_zero_point_input], tensor_dict[y_scale_input], tensor_dict[y_zero_point_input], tensor_dict[B_input_o], tensor_dict[y_output] );
    }

    vuh::Device* QLinearConv::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
