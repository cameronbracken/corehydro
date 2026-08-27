// Tests for Ordinate and LineSimplification (P4 Task 7), the two leaf files of the unported
// Paired Data subsystem (Numerics/Data/Paired Data/).
//
// Oracle is the upstream C# test classes @ 2a0357a:
//   upstream/Numerics/Test_Numerics/Data/Paired Data/Test_Ordinate.cs
//   upstream/Numerics/Test_Numerics/Data/Paired Data/Test_LineSimplification.cs
// Every applicable [TestMethod] is transcribed with its input values copied verbatim.
// Test_ToXElement is SKIPPED: XML round-tripping (the XElement constructor and ToXElement())
// is a project-wide severance -- see ordinate.hpp's header note.
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "corehydro/numerics/data/paired_data/line_simplification.hpp"
#include "corehydro/numerics/data/paired_data/ordinate.hpp"
#include "check.hpp"

using corehydro::numerics::data::SortOrder;
using corehydro::numerics::data::Transform;
using corehydro::numerics::data::paired_data::Ordinate;
namespace line_simplification = corehydro::numerics::data::paired_data::line_simplification;

namespace {

// C# Test_Construction (minus the XElement round trip, which this port does not carry).
void test_construction() {
    Ordinate ordinate1(2, 4);
    Ordinate ordinate3(2, std::numeric_limits<double>::infinity());
    Ordinate ordinate4(std::numeric_limits<double>::quiet_NaN(), 4);

    CHECK_EQ(ordinate1.x, 2.0);
    CHECK_EQ(ordinate1.y, 4.0);

    CHECK_TRUE(ordinate1.is_valid);
    CHECK_TRUE(!ordinate3.is_valid);
    CHECK_TRUE(!ordinate4.is_valid);

    CHECK_TRUE(!(ordinate1 == ordinate3));

    // NaN matches everything in the current set up -- pinned upstream quirk (see ordinate.hpp's
    // transcription note 1): operator== tests fabs(diff) > epsilon, which is false for NaN.
    CHECK_TRUE(ordinate1 == ordinate4);
}

// C# Test_OrdinateValid.
void test_ordinate_valid() {
    Ordinate ordinate(2, 4);
    Ordinate compare1(3, 3);
    Ordinate compare2(3, 5);
    Ordinate compare3(1, 3);
    Ordinate compare4(1, 5);

    bool test1[4] = {
        ordinate.ordinate_valid(compare1, true, true, SortOrder::Ascending, SortOrder::Ascending, true),
        ordinate.ordinate_valid(compare2, true, true, SortOrder::Ascending, SortOrder::Ascending, true),
        ordinate.ordinate_valid(compare3, true, true, SortOrder::Ascending, SortOrder::Ascending, true),
        ordinate.ordinate_valid(compare4, true, true, SortOrder::Ascending, SortOrder::Ascending, true),
    };
    bool test1_expected[4] = {false, true, false, false};
    for (int i = 0; i < 4; ++i) CHECK_EQ(test1[i], test1_expected[i]);

    bool test2[4] = {
        ordinate.ordinate_valid(compare1, true, true, SortOrder::Ascending, SortOrder::Ascending, false),
        ordinate.ordinate_valid(compare2, true, true, SortOrder::Ascending, SortOrder::Ascending, false),
        ordinate.ordinate_valid(compare3, true, true, SortOrder::Ascending, SortOrder::Ascending, false),
        ordinate.ordinate_valid(compare4, true, true, SortOrder::Ascending, SortOrder::Ascending, false),
    };
    bool test2_expected[4] = {false, false, true, false};
    for (int i = 0; i < 4; ++i) CHECK_EQ(test2[i], test2_expected[i]);

    bool test3[4] = {
        ordinate.ordinate_valid(compare1, false, true, SortOrder::None, SortOrder::Ascending, true),
        ordinate.ordinate_valid(compare2, false, true, SortOrder::None, SortOrder::Ascending, true),
        ordinate.ordinate_valid(compare3, false, true, SortOrder::None, SortOrder::Ascending, true),
        ordinate.ordinate_valid(compare4, false, true, SortOrder::None, SortOrder::Ascending, true),
    };
    bool test3_expected[4] = {false, true, false, true};
    for (int i = 0; i < 4; ++i) CHECK_EQ(test3[i], test3_expected[i]);

    bool test4[4] = {
        ordinate.ordinate_valid(compare1, true, false, SortOrder::Ascending, SortOrder::None, true),
        ordinate.ordinate_valid(compare2, true, false, SortOrder::Ascending, SortOrder::None, true),
        ordinate.ordinate_valid(compare3, true, false, SortOrder::Ascending, SortOrder::None, true),
        ordinate.ordinate_valid(compare4, true, false, SortOrder::Ascending, SortOrder::None, true),
    };
    bool test4_expected[4] = {true, true, false, false};
    for (int i = 0; i < 4; ++i) CHECK_EQ(test4[i], test4_expected[i]);

    bool test5[4] = {
        ordinate.ordinate_valid(compare1, false, false, SortOrder::Descending, SortOrder::Ascending, true),
        ordinate.ordinate_valid(compare2, false, false, SortOrder::Descending, SortOrder::Ascending, true),
        ordinate.ordinate_valid(compare3, false, false, SortOrder::Descending, SortOrder::Ascending, true),
        ordinate.ordinate_valid(compare4, false, false, SortOrder::Descending, SortOrder::Ascending, true),
    };
    bool test5_expected[4] = {false, false, false, true};
    for (int i = 0; i < 4; ++i) CHECK_EQ(test5[i], test5_expected[i]);

    bool test6[4] = {
        ordinate.ordinate_valid(compare1, false, false, SortOrder::Ascending, SortOrder::Descending, true),
        ordinate.ordinate_valid(compare2, false, false, SortOrder::Ascending, SortOrder::Descending, true),
        ordinate.ordinate_valid(compare3, false, false, SortOrder::Ascending, SortOrder::Descending, true),
        ordinate.ordinate_valid(compare4, false, false, SortOrder::Ascending, SortOrder::Descending, true),
    };
    bool test6_expected[4] = {true, false, false, false};
    for (int i = 0; i < 4; ++i) CHECK_EQ(test6[i], test6_expected[i]);
}

void check_string_vec(const std::vector<std::string>& actual,
                       const std::vector<std::string>& expected) {
    CHECK_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size() && i < actual.size(); ++i) {
        CHECK_TRUE(actual[i] == expected[i]);
    }
}

// C# Test_OrdinateErrors.
void test_ordinate_errors() {
    Ordinate ordinate(2, 4);
    Ordinate compare2(3, 5);
    Ordinate compare3(1, 3);
    Ordinate compare5(2, 4);

    auto test1 = ordinate.ordinate_errors(compare5, true, true, SortOrder::Ascending,
                                           SortOrder::Descending, true);
    check_string_vec(test1, {"Y values must be strictly decreasing.",
                              "X values must be strictly increasing."});

    auto test2 = ordinate.ordinate_errors(compare2, false, false, SortOrder::Descending,
                                           SortOrder::Descending, true);
    check_string_vec(test2, {"Y values must decrease.", "X values must decrease."});

    auto test3 = ordinate.ordinate_errors(compare3, false, false, SortOrder::Ascending,
                                           SortOrder::Ascending, true);
    check_string_vec(test3, {"Y values must increase.", "X values must increase."});

    auto test4 = ordinate.ordinate_errors(compare3, false, false, SortOrder::Descending,
                                           SortOrder::Descending, false);
    check_string_vec(test4, {"Y values must decrease.", "X values must decrease."});

    auto test5 = ordinate.ordinate_errors(compare2, false, false, SortOrder::Ascending,
                                           SortOrder::Ascending, false);
    check_string_vec(test5, {"Y values must increase.", "X values must increase."});
}

// C# Test_Transform.
void test_transform() {
    Ordinate original(50, 100);
    auto log_x = original.transform(Transform::Logarithmic, Transform::None);
    auto log_y = original.transform(Transform::None, Transform::Logarithmic);
    auto log_xy = original.transform(Transform::Logarithmic, Transform::Logarithmic);

    CHECK_NEAR(log_x.x, 1.69897, 1e-6);
    CHECK_NEAR(log_y.y, 2.0, 1e-6);
    CHECK_NEAR(log_xy.x, 1.69897, 1e-6);
    CHECK_NEAR(log_xy.y, 2.0, 1e-6);

    Ordinate original_z(0.3, 0.8);
    auto z_x = original_z.transform(Transform::NormalZ, Transform::None);
    auto z_y = original_z.transform(Transform::None, Transform::NormalZ);
    auto z_xy = original_z.transform(Transform::NormalZ, Transform::NormalZ);

    CHECK_NEAR(z_x.x, -0.5244005, 1e-6);
    CHECK_NEAR(z_y.y, 0.8416212, 1e-6);
    CHECK_NEAR(z_xy.x, -0.5244005, 1e-6);
    CHECK_NEAR(z_xy.y, 0.8416212, 1e-6);
}

// C# Test_Equality.
void test_equality() {
    Ordinate ordinate1(7.325, 6.389);
    Ordinate ordinate2(7.325, 6.389);
    Ordinate ordinate3(8.36, 25.99);

    bool test[4] = {
        ordinate1 == ordinate2,
        ordinate1 == ordinate3,
        ordinate2 != ordinate3,
        ordinate2 != ordinate1,
    };
    bool expected[4] = {true, false, true, false};
    for (int i = 0; i < 4; ++i) CHECK_EQ(test[i], expected[i]);
}

// C# Test_RamerDouglasPeucker: coordinates of sin(x) on the 3.14 grid.
void test_ramer_douglas_peucker() {
    std::vector<Ordinate> inputs = {
        Ordinate(0, 0),
        Ordinate(3.14 / 2, 1),
        Ordinate(3.14, 0),
        Ordinate(3 * 3.14 / 2, -1),
        Ordinate(2 * 3.14, 0),
    };
    std::vector<Ordinate> outputs;
    double epsilon = 0.1;

    line_simplification::ramer_douglas_peucker(inputs, epsilon, outputs);
    std::vector<Ordinate> valid = {
        Ordinate(0, 0),
        Ordinate(1.57, 1),
        Ordinate(4.71, -1),
        Ordinate(6.28, 0),
    };

    CHECK_EQ(outputs.size(), valid.size());
    for (std::size_t i = 0; i < valid.size() && i < outputs.size(); ++i) {
        CHECK_NEAR(outputs[i].x, valid[i].x, 1e-6);
        CHECK_NEAR(outputs[i].y, valid[i].y, 1e-6);
    }
}

// C# Test_ZeroEpsilon: with epsilon = 0, all points survive.
void test_zero_epsilon() {
    std::vector<Ordinate> inputs = {
        Ordinate(3.6, 8.5),   Ordinate(19.66, 0.33), Ordinate(88.17, 64.9),
        Ordinate(-5.63, 93.2), Ordinate(-22.35, -7.5), Ordinate(-2, -2),
    };
    std::vector<Ordinate> outputs;
    double epsilon = 0;

    line_simplification::ramer_douglas_peucker(inputs, epsilon, outputs);
    CHECK_EQ(outputs.size(), inputs.size());
    for (std::size_t i = 0; i < inputs.size() && i < outputs.size(); ++i) {
        CHECK_TRUE(outputs[i] == inputs[i]);
    }
}

// C# Test_Line: collinear points, only endpoints survive.
void test_line() {
    std::vector<Ordinate> inputs = {
        Ordinate(1, 2), Ordinate(2, 4), Ordinate(3, 6),
        Ordinate(4, 8), Ordinate(5, 10), Ordinate(6, 12),
    };
    std::vector<Ordinate> outputs;
    double epsilon = 0.1;

    line_simplification::ramer_douglas_peucker(inputs, epsilon, outputs);
    CHECK_EQ(outputs.size(), 2u);
    CHECK_TRUE(outputs[0] == inputs.front());
    CHECK_TRUE(outputs[1] == inputs.back());
}

// C# Test_EqualPoints: identical points -- the mag > 0.0 degenerate-segment guard case.
void test_equal_points() {
    std::vector<Ordinate> inputs = {
        Ordinate(1, 30), Ordinate(1, 30), Ordinate(1, 30),
        Ordinate(1, 30), Ordinate(1, 30), Ordinate(1, 30),
    };
    std::vector<Ordinate> outputs;
    double epsilon = 0.1;

    line_simplification::ramer_douglas_peucker(inputs, epsilon, outputs);
    CHECK_EQ(outputs.size(), 2u);
    CHECK_TRUE(outputs[0] == inputs.front());
    CHECK_TRUE(outputs[1] == inputs.back());
}

}  // namespace

int main() {
    test_construction();
    test_ordinate_valid();
    test_ordinate_errors();
    test_transform();
    test_equality();
    test_ramer_douglas_peucker();
    test_zero_epsilon();
    test_line();
    test_equal_points();
    return chtest::summary("test_ordinate");
}
