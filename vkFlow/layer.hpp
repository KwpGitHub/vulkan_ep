#pragma once
#ifndef LAYER_H
#define LAYER_H

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
		device = instance.devices().at(deviceID);
	}

	layer() : deviceID(0), device(instance.devices().at(deviceID)) {

	}
	~layer(){}

	virtual void foward(){}

	void backward() {}
	
protected:
	kernel::Instance instance = kernel::Instance();
	kernel::Device device;
	uint32_t deviceID;
	size_t N;
};

#endif  //!LAYER_H

