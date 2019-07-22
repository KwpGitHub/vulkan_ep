#include <vector>
namespace backend {
    class LRN {
    public:
        LRN ();
        ~LRN();
    private:

		float alpha;
		float beta;
		float bias;
		int size;
    };
}
