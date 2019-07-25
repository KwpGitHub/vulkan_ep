#include <cstdio>
#include <map>
#include <vector>
#include "device.h"
#include "allocator.h"
#include "command.h"
#include "layer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <iomanip>

namespace py = pybind11;


static backend::Device* vkdev = 0;
static backend::Allocator* blob_allocator = 0;
static backend::Allocator* staging_allocator = 0;
static std::vector<backend::Layer*> layers;

void create_device() {
	backend::create_gpu_instance();
	vkdev = backend::get_gpu_device(1);
	blob_allocator = new backend::BlobBufferAllocator(vkdev);
	staging_allocator = new backend::StagingBufferAllocator(vkdev);

	auto info = vkdev->info;
	std::cout << std::left << std::setw(64) << "name" << std::setw(20) << "data" << std::endl;
	std::cout << std::string(64, '-') << std::endl;
	std::cout << std::left << std::setw(64) << "device type" << std::setw(20) << info.type << std::endl;
	std::cout << std::left << std::setw(64) << "api_version" << std::setw(20) << info.api_version << std::endl;

	std::cout << std::left << std::setw(64) << "driver_version" << std::setw(20) << info.driver_version << std::endl;
	std::cout << std::left << std::setw(64) << "vendor_id" << std::setw(20) << info.vendor_id << std::endl;
	std::cout << std::left << std::setw(64) << "device_id" << std::setw(20) << info.device_id << std::endl;
	std::cout << std::left << std::setw(64) << "max_shared_memory_size" << std::setw(20) << info.max_shared_memory_size << std::endl;
	std::cout << std::left << std::setw(64) << "max_workgroup_invocations" << std::setw(20) << info.max_workgroup_invocations << std::endl;

	std::cout << std::left << std::setw(64) << "max_workgroup_count" << std::setw(16) << info.max_workgroup_count[0] << std::setw(8) << info.max_workgroup_count[1] << std::setw(8) << info.max_workgroup_count[2] << std::endl;
	std::cout << std::left << std::setw(64) << "max_workgroup_size" << std::setw(8) << info.max_workgroup_size[0] << std::setw(8) << info.max_workgroup_size[1] << std::setw(8) << info.max_workgroup_size[2] << std::endl;

	std::cout << std::left << std::setw(64) << "memory_map_alignment" << std::setw(20) << info.memory_map_alignment << std::endl;
	std::cout << std::left << std::setw(64) << "buffer_offset_alignment" << std::setw(20) << info.buffer_offset_alignment << std::endl;

}



void run() {
	{
		backend::VkTransfer cmd(vkdev);
		backend::VkCompute cmpt(vkdev);
		blob_allocator->clear();
		staging_allocator->clear();
	}
	backend::destroy_gpu_instance();
}



PYBIND11_MODULE(onnx_ep, m) {
	m.def("run", &run);
	m.def("create_device", &create_device);
}