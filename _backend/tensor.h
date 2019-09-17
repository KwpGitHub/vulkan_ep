#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <numeric>
#include <map>

#include "kernel/array.hpp"
#include "kernel/vuh.h"

namespace backend {
	inline vuh::Instance* g_instance;
	inline vuh::Device* g_device;
}

namespace backend {

	struct Shape_t {
		uint32_t n;
		uint32_t c;
		uint32_t d;
		uint32_t h;
		uint32_t w;
	};

	class Tensor {		
	public:
		std::string name;
		vuh::Device* dev;
		vuh::Array<float>* data;
		Shape_t dims;
		size_t size;

		Tensor(): data(nullptr), size(0u), dev(nullptr) {}
		
		Tensor(const std::vector<float>& d, Shape_t s): dims(s) {
			dev = g_device;
			size = (size_t)dims.n * (size_t)dims.c * (size_t)dims.d * (size_t)dims.h * (size_t)dims.w;
			data = new vuh::Array<float>(*dev, begin(d), end(d));
		}
		
		Tensor(const Tensor& t) {			
			data = t.data;			
			dims = t.dims;
			size = t.size;
			dev = t.dev;
		}

		std::vector<float> to_vector() {
			std::vector<float> t(size, 1.0);
			data->toHost(begin(t));
			return t;
		}

		void to(int d) {			
			std::vector<float> t(size, 0.0);
			data->toHost(begin(t));
			delete data;			
			data = new vuh::Array<float>(g_instance->devices().at(d), t);
		}

		void to(vuh::Device* d) {
			std::vector<float> t(size, 0.0);
			data->toHost(begin(t));
			delete data;
			data = new vuh::Array<float>(*d, t);
		}

		Shape_t shape() {
			return dims;
		}

		~Tensor() {
			delete data;			
		}

	};
}


namespace backend {
	inline std::map<std::string, Tensor*> tensor_dict;
	inline std::string file_path;
}

#endif





#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
enum Format {
	kFormatInvalid = -1,
	kFormatFp16,
	kFormatFp32,
	kFormatFp64,
	kFormatInt32,
	kFormatNum
};

class Buffer
{
public:
	Buffer(VkDevice& device) : device_(device), buffer_(VK_NULL_HANDLE), memory_(VK_NULL_HANDLE) {};
	Buffer(VkDevice& device, size_t size_in_bytes, const char* data);
	~Buffer();
	VkDeviceMemory getVkMemory() { return memory_; }
	VkBuffer getVkBuffer() { return buffer_; }

private:
	Buffer();
	bool init(size_t size_in_bytes, const char* data);
	VkDevice device_;
	VkBuffer buffer_;
	VkDeviceMemory memory_;
};

class Tensor
{
public:
	Tensor(Format fmt = kFormatFp32);
	Tensor(const char* data, std::vector<int>& shape, Format fmt = kFormatFp32);
	void* map();
	void unMap();
	Shape getShape() const;
	int dimSize(const int dim) const;
	int dimNum() const;
	int count(const int start_axis = 0, const int end_axis = -1) const;

	// Change shape and format to as passed in.
	// Copy data if data != NULL
	// Allocate new internal buffer if new size > old size or alloc flag is true
	Tensor reshape(const char* data, const std::vector<int>& shape, bool alloc = false, Format fmt = kFormatInvalid);

	void setTo(float val);
	int getFormat() const;
	size_t size() const { return size_in_byte_; }
	bool isEmpty() { return size_in_byte_ == 0 ? true : false; }
	void copyTo(Tensor& dst);
	std::shared_ptr<Buffer> getBuffer() { return buffer_; }

private:
	VkDevice device_;
	std::vector<int> shape_;
	size_t size_in_byte_;
	std::shared_ptr<Buffer> buffer_;
	Format format_;
};

#endif  // HAVE_VULKAN