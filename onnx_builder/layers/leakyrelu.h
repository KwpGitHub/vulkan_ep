#include <vector>
namespace backend {
    class LeakyRelu {
    public:
        LeakyRelu ();
        ~LeakyRelu();
    private:

		float alpha;
    };
}
