#ifndef TILE_LAYER_H
#define TILE_LAYER_H
#include "../layer.h"

namespace backend {
	namespace CPU {
		class Tile : public Layer {
		public:
			Tile();
			virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

			int dim;
			int tiles;
		};
	}
	namespace GPU {
		class Tile : virtual public CPU::Tile {

		public:
			Tile();
		};
	}
}

#endif

