#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "corehydro/numerics/support/callback_guard.hpp"

namespace sup = corehydro::numerics::support;

struct HostError : std::runtime_error {
    HostError() : std::runtime_error("host language error") {}
};

int main() {
    // A guard that never sees a throw is transparent and counts real calls.
    {
        sup::GuardedCall<double, const std::vector<double>&> g(
            [](const std::vector<double>& p) { return p[0] * 2.0; }, -1.0);
        CHECK_EQ(g({3.0}), 6.0);
        CHECK_EQ(g({4.0}), 8.0);
        CHECK_TRUE(!g.aborted());
        CHECK_EQ(g.call_count(), 2);
    }

    // The first throw latches, later calls short-circuit to the sentinel WITHOUT re-entering
    // the host function, and the stored exception rethrows with its original type.
    {
        int entries = 0;
        sup::GuardedCall<double, const std::vector<double>&> g(
            [&entries](const std::vector<double>&) -> double {
                ++entries;
                throw HostError();
            },
            -1.0);
        CHECK_EQ(g({1.0}), -1.0);
        CHECK_EQ(g({2.0}), -1.0);
        CHECK_EQ(entries, 1);         // second call never reached the host
        CHECK_EQ(g.call_count(), 0);  // no call completed
        CHECK_TRUE(g.aborted());
        CHECK_THROWS_MSG(g.rethrow_if_aborted(), "host language error");
    }

    // A vector-returning guard (the shape the Gibbs proposal and bootstrap resample need).
    {
        sup::GuardedCall<std::vector<double>, const std::vector<double>&> g(
            [](const std::vector<double>& p) { return std::vector<double>{p[0], p[0]}; },
            std::vector<double>{});
        CHECK_EQ(g({5.0}).size(), std::size_t{2});
        CHECK_TRUE(!g.aborted());
    }

    return chtest::summary("callback_runner");
}
