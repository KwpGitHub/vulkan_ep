#pragma once
#include "kernel/kernel.h"
#include "kernel/array.hpp"
#include "utils.hpp"
#include <vector>


class layer
{
public:
	virtual void to(int id)
	{
		deviceID = id;
		device = runtime_info::instance.devices().at(deviceID);
	}

	layer() : deviceID(0), device(runtime_info::instance.devices().at(deviceID)) {

	}
	~layer(){}

	virtual void run(){}
	
protected:
	kernel::Device device;
	uint32_t deviceID;
};

