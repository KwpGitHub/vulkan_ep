#include <vector>
namespace backend {
    class Multinomial {
    public:
        Multinomial ();
        ~Multinomial();
    private:

		int dtype;
		int sample_size;
		float seed;
    };
}
