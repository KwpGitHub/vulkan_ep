#ifndef GEMM_H
#define GEMM_H //Gemm

//INPUTS:                   A, B, C
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, beta, transA, transB
//OPTIONAL_PARAMETERS_TYPE: float, float, int, int



namespace backend {
    class Gemm : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            float alpha; float beta; int transA; int transB;
			
            //input
            Shape_t A; Shape_t B; Shape_t C;
            
            //output
            Shape_t Y;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Gemm(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        float alpha; float beta; int transA; int transB;
		
        //input
        std::string A; std::string B; std::string C;
        
        //output
        std::string Y;
        
        //std::vector<uint32_t> output_shape();
   
        ~Gemm(){}
    };
}


namespace backend {    
    Gemm::Gemm(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/gemm.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* Gemm::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
