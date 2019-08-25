#include "sum.h"
//cpp stuff
namespace layers {    
   
    Sum::Sum(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/sum.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Sum::_get_device() {
        
        return backend::device;
    }
    
    void Sum::init() {      
  
    }
    
    void Sum::bind(std::string _x0_i, std::string _x1_i, std::string _x2_i, std::string _x3_i, std::string _x4_i, std::string _x5_i, std::string _x6_i, std::string _x7_i, std::string _x8_i, std::string _x9_i, std::string _x10_i, std::string _x11_i, std::string _x12_i, std::string _x13_i, std::string _x14_i, std::string _x15_i, std::string _x16_i, std::string _x17_i, std::string _x18_i, std::string _x19_i, std::string _x20_i, std::string _x21_i, std::string _x22_i, std::string _x23_i, std::string _x24_i, std::string _x25_i, std::string _x26_i, std::string _x27_i, std::string _x28_i, std::string _x29_i, std::string _x30_i, std::string _x31_i, std::string _sum_o){
        x0_i = _x0_i; x1_i = _x1_i; x2_i = _x2_i; x3_i = _x3_i; x4_i = _x4_i; x5_i = _x5_i; x6_i = _x6_i; x7_i = _x7_i; x8_i = _x8_i; x9_i = _x9_i; x10_i = _x10_i; x11_i = _x11_i; x12_i = _x12_i; x13_i = _x13_i; x14_i = _x14_i; x15_i = _x15_i; x16_i = _x16_i; x17_i = _x17_i; x18_i = _x18_i; x19_i = _x19_i; x20_i = _x20_i; x21_i = _x21_i; x22_i = _x22_i; x23_i = _x23_i; x24_i = _x24_i; x25_i = _x25_i; x26_i = _x26_i; x27_i = _x27_i; x28_i = _x28_i; x29_i = _x29_i; x30_i = _x30_i; x31_i = _x31_i; sum_o = _sum_o;

		//binding.x0_i = tensor_dict[x0_i]->shape();
  		//binding.x1_i = tensor_dict[x1_i]->shape();
  		//binding.x2_i = tensor_dict[x2_i]->shape();
  		//binding.x3_i = tensor_dict[x3_i]->shape();
  		//binding.x4_i = tensor_dict[x4_i]->shape();
  		//binding.x5_i = tensor_dict[x5_i]->shape();
  		//binding.x6_i = tensor_dict[x6_i]->shape();
  		//binding.x7_i = tensor_dict[x7_i]->shape();
  		//binding.x8_i = tensor_dict[x8_i]->shape();
  		//binding.x9_i = tensor_dict[x9_i]->shape();
  		//binding.x10_i = tensor_dict[x10_i]->shape();
  		//binding.x11_i = tensor_dict[x11_i]->shape();
  		//binding.x12_i = tensor_dict[x12_i]->shape();
  		//binding.x13_i = tensor_dict[x13_i]->shape();
  		//binding.x14_i = tensor_dict[x14_i]->shape();
  		//binding.x15_i = tensor_dict[x15_i]->shape();
  		//binding.x16_i = tensor_dict[x16_i]->shape();
  		//binding.x17_i = tensor_dict[x17_i]->shape();
  		//binding.x18_i = tensor_dict[x18_i]->shape();
  		//binding.x19_i = tensor_dict[x19_i]->shape();
  		//binding.x20_i = tensor_dict[x20_i]->shape();
  		//binding.x21_i = tensor_dict[x21_i]->shape();
  		//binding.x22_i = tensor_dict[x22_i]->shape();
  		//binding.x23_i = tensor_dict[x23_i]->shape();
  		//binding.x24_i = tensor_dict[x24_i]->shape();
  		//binding.x25_i = tensor_dict[x25_i]->shape();
  		//binding.x26_i = tensor_dict[x26_i]->shape();
  		//binding.x27_i = tensor_dict[x27_i]->shape();
  		//binding.x28_i = tensor_dict[x28_i]->shape();
  		//binding.x29_i = tensor_dict[x29_i]->shape();
  		//binding.x30_i = tensor_dict[x30_i]->shape();
  		//binding.x31_i = tensor_dict[x31_i]->shape();
 
		//binding.sum_o = tensor_dict[sum_o]->shape();
 
        
    }

    void Sum::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[x0_i]->data(), *tensor_dict[x1_i]->data(), *tensor_dict[x2_i]->data(), *tensor_dict[x3_i]->data(), *tensor_dict[x4_i]->data(), *tensor_dict[x5_i]->data(), *tensor_dict[x6_i]->data(), *tensor_dict[x7_i]->data(), *tensor_dict[x8_i]->data(), *tensor_dict[x9_i]->data(), *tensor_dict[x10_i]->data(), *tensor_dict[x11_i]->data(), *tensor_dict[x12_i]->data(), *tensor_dict[x13_i]->data(), *tensor_dict[x14_i]->data(), *tensor_dict[x15_i]->data(), *tensor_dict[x16_i]->data(), *tensor_dict[x17_i]->data(), *tensor_dict[x18_i]->data(), *tensor_dict[x19_i]->data(), *tensor_dict[x20_i]->data(), *tensor_dict[x21_i]->data(), *tensor_dict[x22_i]->data(), *tensor_dict[x23_i]->data(), *tensor_dict[x24_i]->data(), *tensor_dict[x25_i]->data(), *tensor_dict[x26_i]->data(), *tensor_dict[x27_i]->data(), *tensor_dict[x28_i]->data(), *tensor_dict[x29_i]->data(), *tensor_dict[x30_i]->data(), *tensor_dict[x31_i]->data(), *tensor_dict[sum_o]->data());
    }

}

