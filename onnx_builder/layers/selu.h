#include <vector>
namespace backend {
    class Selu {
    public:
        Selu ();
        ~Selu();
    private:

		float alpha;
		float gamma;
    };
}
