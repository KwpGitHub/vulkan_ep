#ifndef QLINEARMATMUL_H
#define QLINEARMATMUL_H //QLinearMatMul

//INPUTS:                   a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
//OPTIONAL_INPUTS:          
//OUTPUS:                   y
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
            Shape_t a; Shape_t a_scale; Shape_t a_zero_point; Shape_t b; Shape_t b_scale; Shape_t b_zero_point; Shape_t y_scale; Shape_t y_zero_point;
            
            //output
            Shape_t y;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        QLinearMatMul(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        
		
        //input
        std::string a; std::string a_scale; std::string a_zero_point; std::string b; std::string b_scale; std::string b_zero_point; std::string y_scale; std::string y_zero_point;
        
        //output
        std::string y;
        
        //std::vector<uint32_t> output_shape();
   
        ~QLinearMatMul(){}
    };
}


namespace backend {    
    QLinearMatMul::QLinearMatMul(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/qlinearmatmul.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* QLinearMatMul::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
