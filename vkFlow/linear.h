#pragma once
#include "layer.hpp"

class linear :
	public layer
{
public:
	linear(int in_channel, int out_channel, bool usebias = true) {

	}

	void forward() {}

	void backward() {}

protected: //y = x1 * AT + b
	int in_features;
	int out_features;

	Shape weight_shape;
	Shape bias_hsape;
	std::vector<float> weight;
	std::vector<float> bias;
};

class bilinear : //y = x1 * A * x2 + b
	public layer
{
public:
	bilinear(int in_channel, int in2_channel, int out_channel, bool usebias = true) {

	}

	void forward() {}
	
	void backward() {}

protected:
	int in1_features;
	int in2_features;
	int out_features;

	Shape weight_shape;
	Shape bias_hsape;
	std::vector<float> weight;
	std::vector<float> bias;
};

