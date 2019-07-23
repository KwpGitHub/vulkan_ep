#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "device.h"
#include "allocator.h"

namespace py = pybind11;


static backend::Device* vkdev = 0;
static backend::Allocator* blob_allocator = 0;
static backend::Allocator* staging_allocator = 0;


void create_device() {
	backend::create_gpu_instance();
	vkdev = backend::get_gpu_device(1);
	blob_allocator = new backend::BlobBufferAllocator(vkdev);
	staging_allocator = new backend::StagingBufferAllocator(vkdev);
}



void run() {
	vkdev->info;
}



PYBIND11_MODULE(onnx_ep, m) {
	m.def("run", &run);
	m.def("create_device", &create_device);
}