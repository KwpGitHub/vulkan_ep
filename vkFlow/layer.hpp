#pragma once
#include "pipeline/pipeline.h"
#include "pipeline/array.hpp"
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
	pipeline::Device device;
	uint32_t deviceID;
};

