#ifndef QLINEARCONV_H
#define QLINEARCONV_H //QLinearConv

//INPUTS:                   x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point
//OPTIONAL_INPUTS:          B
//OUTPUS:                   y
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
            Shape_t x; Shape_t x_scale; Shape_t x_zero_point; Shape_t w; Shape_t w_scale; Shape_t w_zero_point; Shape_t y_scale; Shape_t y_zero_point;
            Shape_t B;
            //output
            Shape_t y;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        QLinearConv(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int auto_pad; Shape_t dilations; int group; Shape_t kernel_shape; Shape_t pads; Shape_t strides;
		
        //input
        std::string x; std::string x_scale; std::string x_zero_point; std::string w; std::string w_scale; std::string w_zero_point; std::string y_scale; std::string y_zero_point;
        std::string B;
        //output
        std::string y;
        
        //std::vector<uint32_t> output_shape();
   
        ~QLinearConv(){}
    };
}


namespace backend {    
    QLinearConv::QLinearConv(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/qlinearconv.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* QLinearConv::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
