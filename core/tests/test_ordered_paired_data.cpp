// Tests for OrderedPairedData (P4 Task 8) and the six un-severed Search.cs overloads it un-gates.
//
// Oracle is the upstream C# test classes @ 2a0357a:
//   upstream/Numerics/Test_Numerics/Data/Paired Data/Test_PairedData.cs
//   upstream/Numerics/Test_Numerics/Data/Paired Data/Test_PairedDataInterpolation.cs
//   upstream/Numerics/Test_Numerics/Data/Paired Data/Test_PairedDataLineSimplification.cs
// Every applicable [TestMethod] is transcribed with its input values copied verbatim.
// Test_ReadWriteXElement is SKIPPED: XML round-tripping is a project-wide severance (see
// ordered_paired_data.hpp's header note).
//
// Test_Indexing's CopyTo(Ordinate[], int) segment is also skipped: CopyTo is plain
// ICollection<T> boilerplate (a straight array copy) not in this port's member surface (see
// ordered_paired_data.hpp's header note on what is/isn't ported); the assertion it backs
// (`dataset11[i] == array[i]`) is definitionally true of any correct copy and tests nothing
// beyond what the surrounding indexing assertions already cover, so no equivalent is
// substituted.
#include <cmath>
#include <vector>

#include "corehydro/numerics/data/interpolation/search.hpp"
#include "corehydro/numerics/data/paired_data/ordered_paired_data.hpp"
#include "check.hpp"

using corehydro::numerics::data::SortOrder;
using corehydro::numerics::data::Transform;
using corehydro::numerics::data::paired_data::Ordinate;
using corehydro::numerics::data::paired_data::OrderedPairedData;
namespace search = corehydro::numerics::data::search;

namespace {

std::vector<double> reversed(const std::vector<double>& v) {
    return std::vector<double>(v.rbegin(), v.rend());
}

// -------------------------------------------------------------------------------------------
// Shared 15-point reservoir fixture (Test_PairedData's constructor), transcribed verbatim.
// -------------------------------------------------------------------------------------------
const std::vector<double> kCtorX = {
    230408, 288010, 345611, 403213, 460815, 518417, 576019, 633612,
    691223, 748825, 806427, 864029, 921631, 1036834, 1152038,
};

const std::vector<double> kCtorY = {
    1519.7, 1520.5, 1520.9, 1521.7, 1523.5, 1525.9, 1528.4, 1530.9,
    1533.2, 1534.7, 1535.9, 1538, 1541.3, 1547.7, 1552.7,
};

// Test_GetY / Test_GetX's 78-element query and expected arrays, transcribed verbatim.
const std::vector<double> kQueryX = {
    1018627, 742619,  770076,  350167,  260164,  502421,  1034555, 810438,
    655158,  253951,  1149424, 973525,  397450,  1128872, 640330,  303494,
    668286,  373493,  731302,  518190,  553336,  897626,  682656,  580591,
    915612,  949596,  411371,  660301,  793095,  651386,  289081,  958856,
    346709,  523205,  930560,  500896,  1056651, 581860,  653012,  1000323,
    592979,  1042845, 418936,  494984,  447825,  653758,  250286,  233175,
    532844,  567222,  508867,  941194,  607007,  275898,  565432,  654433,
    1138135, 465458,  433568,  983439,  765617,  810132,  570295,  1135444,
    1039781, 346336,  851548,  607131,  312623,  836800,  348436,  756059,
    275085,  903306,  955898,  374842,  597348,  741079,
};

// Expected Y for datasets 1 and 4 (ascending/ascending and descending/descending).
const std::vector<double> kExpectedY14 = {
    1546.68852634046, 1534.53839102809, 1535.14271379466, 1520.96327558071, 1520.11326342835,
    1525.23352314156, 1547.57339218597, 1536.04622929759, 1531.76017947961, 1520.02697475782,
    1552.58654907816, 1544.18292492383, 1521.61996111246, 1551.6945661609,  1531.16820225304,
    1520.60752591101, 1532.28428772283, 1521.28723655429, 1534.24368771918, 1525.89054199507,
    1527.41552897469, 1539.92476129301, 1532.85798024683, 1528.5984616186,  1540.95517343148,
    1542.85357065354, 1521.95492864831, 1531.96550311572, 1535.62225964376, 1531.60959018243,
    1520.50743737088, 1543.36800170134, 1520.9152494705,  1526.10780528454, 1541.79604263778,
    1525.16998368112, 1548.56008298323, 1528.65354643794, 1531.67450486886, 1545.67166393236,
    1529.13620057993, 1547.96088503871, 1522.19132669005, 1524.92365890073, 1523.09407659456,
    1531.70428737567, 1519.97607374744, 1519.73842922121, 1526.52615013368, 1528.01819902087,
    1525.5020971494,  1542.38680503112, 1529.74512874829, 1520.33178361862, 1527.94051074615,
    1531.73123535436, 1552.09659213222, 1523.69345161626, 1522.64856081386, 1544.73368835881,
    1535.04982118676, 1536.03507343495, 1528.151571126,   1551.97979931252, 1547.82790354502,
    1520.91006909482, 1537.544979341,   1529.75051134687, 1520.67092064374, 1537.00731051005,
    1520.93923474879, 1534.85070310059, 1520.32049234402, 1540.25016666088, 1543.20367264741,
    1521.30597201486, 1529.32585036376, 1534.49828825388,
};

// Expected Y for datasets 2 and 3 (descending-x/ascending-y and ascending-x/descending-y).
const std::vector<double> kExpectedY23 = {
    1520.56321710372, 1526.16934828652, 1525.01457241068, 1541.03898822958, 1550.11710357279,
    1535.03323842922, 1520.50791298838, 1523.37466060206, 1529.96502230477, 1550.65640776362,
    1519.71815214749, 1520.71981719226, 1538.3301604111,  1519.86086941426, 1530.60847581191,
    1545.97958542386, 1529.39533943171, 1539.70264921357, 1526.66052046804, 1534.70472900247,
    1533.79068261519, 1521.23339120169, 1528.77176060127, 1533.01741531089, 1520.98359431964,
    1520.80290183415, 1537.70258324364, 1529.74184443943, 1524.05548071248, 1530.12870632345,
    1547.58100206594, 1520.77074989367, 1541.23709593417, 1534.57531682928, 1520.86899733514,
    1535.06500815944, 1520.36238672268, 1532.9667372771,  1530.05814688167, 1520.62677100423,
    1532.52269546646, 1520.45825839381, 1537.42678552828, 1535.18817054963, 1536.37357730634,
    1530.02577459166, 1550.9745390785,  1552.45981736745, 1534.32430991979, 1533.42908058748,
    1534.8989514253,  1520.83207468556, 1531.96248155158, 1548.7513523836,  1533.47569355231,
    1529.99648331048, 1519.79654525885, 1535.80327419187, 1536.89334571716, 1520.68539447757,
    1525.20035762647, 1523.38422277004, 1533.3490573244,  1519.81523211,    1520.4795354328,
    1541.25846498385, 1522.09001770772, 1531.95752956088, 1544.96526970018, 1522.55087670567,
    1541.13815666123, 1525.59859379883, 1548.8219228499,  1521.15450505191, 1520.78102045954,
    1539.6253654387,  1532.34821766534, 1526.23618624353,
};

struct Datasets {
    OrderedPairedData d1, d2, d3, d4;
};

Datasets build_datasets() {
    return Datasets{
        OrderedPairedData(kCtorX, kCtorY, true, SortOrder::Ascending, true, SortOrder::Ascending),
        OrderedPairedData(reversed(kCtorX), kCtorY, true, SortOrder::Descending, true,
                           SortOrder::Ascending),
        OrderedPairedData(kCtorX, reversed(kCtorY), true, SortOrder::Ascending, true,
                           SortOrder::Descending),
        OrderedPairedData(reversed(kCtorX), reversed(kCtorY), true, SortOrder::Descending, true,
                           SortOrder::Descending),
    };
}

void sample_dataset_y(const OrderedPairedData& data, const std::vector<double>& samples,
                       const std::vector<double>& expected) {
    for (std::size_t i = 0; i < samples.size(); ++i)
        CHECK_NEAR(data.get_y_from_x(samples[i]), expected[i], 1e-5);
    auto from_list = data.get_y_from_x(samples);
    for (std::size_t i = 0; i < samples.size(); ++i) CHECK_NEAR(from_list[i], expected[i], 1e-5);
}

// C# Test_GetY.
void test_get_y() {
    auto ds = build_datasets();
    sample_dataset_y(ds.d1, kQueryX, kExpectedY14);
    sample_dataset_y(ds.d4, kQueryX, kExpectedY14);
    sample_dataset_y(ds.d2, kQueryX, kExpectedY23);
    sample_dataset_y(ds.d3, kQueryX, kExpectedY23);
}

void sample_dataset_x(const OrderedPairedData& data, const std::vector<double>& samples,
                       const std::vector<double>& expected) {
    for (std::size_t i = 0; i < samples.size(); ++i)
        CHECK_NEAR(data.get_x_from_y(samples[i]), expected[i], 1e-5);
    auto from_list = data.get_x_from_y(samples);
    for (std::size_t i = 0; i < samples.size(); ++i) CHECK_NEAR(from_list[i], expected[i], 1e-5);
}

// C# Test_GetX: same three arrays as Test_GetY, roles swapped (samples = the expected-Y arrays,
// expected = the shared query array).
void test_get_x() {
    auto ds = build_datasets();
    sample_dataset_x(ds.d1, kExpectedY14, kQueryX);
    sample_dataset_x(ds.d4, kExpectedY14, kQueryX);
    sample_dataset_x(ds.d2, kExpectedY23, kQueryX);
    sample_dataset_x(ds.d3, kExpectedY23, kQueryX);
}

// C# Test_Indexing.
void test_indexing() {
    auto ds = build_datasets();
    OrderedPairedData dataset11 = ds.d1.clone();
    Ordinate ordinate(460815, 1523.5);

    Ordinate test1 = dataset11[4];
    CHECK_TRUE(ordinate == test1);

    int test2 = dataset11.index_of(ordinate);
    int test3 = dataset11.index_of(460815, 1523.5);
    CHECK_EQ(test2, 4);
    CHECK_EQ(test3, 4);

    dataset11.remove(ordinate);
    CHECK_TRUE(!dataset11.contains(ordinate));

    Ordinate new_ordinate = dataset11[4];
    dataset11.remove_at(4);
    CHECK_TRUE(!dataset11.contains(new_ordinate));

    Ordinate new_ordinate2(1243177, 1563.8);
    dataset11.add(new_ordinate2);
    int test6 = dataset11.index_of(new_ordinate2);
    CHECK_EQ(test6, dataset11.count() - 1);

    dataset11.insert(4, ordinate);
    int test7 = dataset11.index_of(ordinate);
    CHECK_EQ(test7, 4);

    int prev_count = dataset11.count();
    Ordinate new_ordinate3 = dataset11[3];
    dataset11.remove_range(0, 3);
    Ordinate test8 = dataset11[0];
    int curr_count = dataset11.count();
    CHECK_EQ(curr_count, prev_count - 3);
    CHECK_TRUE(new_ordinate3 == test8);

    auto inverted = dataset11.invert();
    for (int i = 0; i < dataset11.count(); ++i) {
        CHECK_EQ(dataset11[i].x, inverted[i].y);
        CHECK_EQ(dataset11[i].y, inverted[i].x);
    }

    dataset11.clear();
    CHECK_EQ(dataset11.count(), 0);
}

// C# Test_Equality.
void test_equality() {
    auto ds = build_datasets();
    OrderedPairedData dataset5 = ds.d1.clone();
    CHECK_TRUE(ds.d1 == dataset5);
    CHECK_TRUE(!(ds.d1 == ds.d2));
    CHECK_TRUE(ds.d2 != ds.d3);
    CHECK_TRUE(!(ds.d1 != dataset5));
}

// C# Test_TrapezoidalArea.
void test_trapezoidal_area() {
    auto ds = build_datasets();
    CHECK_NEAR(ds.d1.trapezoidal_area_under_y(), 1413175623.0, 1.0);
    CHECK_NEAR(ds.d1.trapezoidal_area_under_x(), 25442742.0, 1.0);
    CHECK_NEAR(ds.d2.trapezoidal_area_under_y(), 1410070832.0, 1.0);
    CHECK_NEAR(ds.d2.trapezoidal_area_under_x(), 17073185.0, 1.0);
    CHECK_NEAR(ds.d3.trapezoidal_area_under_y(), 1410070832.0, 1.0);
    CHECK_NEAR(ds.d3.trapezoidal_area_under_x(), 17073185.0, 1.0);
    CHECK_NEAR(ds.d4.trapezoidal_area_under_y(), 1413175623.0, 1.0);
    CHECK_NEAR(ds.d4.trapezoidal_area_under_x(), 25442742.0, 1.0);
}

// Shared 1000-point identity-curve fixture for Test_Sequential/_Bisection/_Hunt.
OrderedPairedData build_identity_ascending() {
    OrderedPairedData opd(true, SortOrder::Ascending, false, SortOrder::Ascending);
    for (int i = 1; i <= 1000; ++i) opd.add(Ordinate(i, i));
    return opd;
}

OrderedPairedData build_identity_descending() {
    OrderedPairedData opd(true, SortOrder::Descending, false, SortOrder::Descending);
    for (int i = 1000; i >= 1; --i) opd.add(Ordinate(i, i));
    return opd;
}

// C# Test_Sequential.
void test_sequential() {
    auto asc = build_identity_ascending();
    CHECK_EQ(asc.sequential_search_x(872.5), 871);
    CHECK_EQ(asc.sequential_search_y(872.5), 871);

    auto dsc = build_identity_descending();
    CHECK_EQ(dsc.sequential_search_x(872.5), 127);
    CHECK_EQ(dsc.sequential_search_y(872.5), 127);
}

// C# Test_Bisection.
void test_bisection() {
    auto asc = build_identity_ascending();
    CHECK_EQ(asc.bisection_search_x(872.5), 871);
    CHECK_EQ(asc.bisection_search_y(872.5), 871);

    auto dsc = build_identity_descending();
    CHECK_EQ(dsc.bisection_search_x(872.5), 127);
    CHECK_EQ(dsc.bisection_search_y(872.5), 127);
}

// C# Test_Hunt.
void test_hunt() {
    auto asc = build_identity_ascending();
    CHECK_EQ(asc.hunt_search_x(872.5), 871);
    CHECK_EQ(asc.hunt_search_y(872.5), 871);

    auto dsc = build_identity_descending();
    CHECK_EQ(dsc.hunt_search_x(872.5), 127);
    CHECK_EQ(dsc.hunt_search_y(872.5), 127);
}

// Corehydro addition: cross-check the newly-ported search::sequential/bisection/hunt free
// functions (Search.cs's OrderedPairedData/Ordinate overloads, C# lines 167/254/444/545/782/925)
// against OrderedPairedData's own member search methods, on the same 1000-point curve the C#
// suite uses above. Not an upstream test -- Search.cs has no test file of its own for these
// overloads -- but required by the task brief to prove the un-severed free functions agree with
// the class's own search machinery.
void test_search_cross_check() {
    auto asc = build_identity_ascending();
    CHECK_EQ(search::sequential(872.5, asc), asc.sequential_search_x(872.5));
    CHECK_EQ(search::bisection(872.5, asc), asc.bisection_search_x(872.5));
    CHECK_EQ(search::hunt(872.5, asc), asc.hunt_search_x(872.5));

    auto dsc = build_identity_descending();
    CHECK_EQ(search::sequential(872.5, dsc), dsc.sequential_search_x(872.5));
    CHECK_EQ(search::bisection(872.5, dsc), dsc.bisection_search_x(872.5));
    CHECK_EQ(search::hunt(872.5, dsc), dsc.hunt_search_x(872.5));

    // The vector<Ordinate> overloads take the ordinates directly rather than an OrderedPairedData,
    // so build the same 1000-point curves as plain vectors.
    std::vector<Ordinate> asc_ords, dsc_ords;
    for (int i = 1; i <= 1000; ++i) asc_ords.emplace_back(i, i);
    for (int i = 1000; i >= 1; --i) dsc_ords.emplace_back(i, i);

    CHECK_EQ(search::sequential(872.5, asc_ords, 0, SortOrder::Ascending), asc.sequential_search_x(872.5));
    CHECK_EQ(search::bisection(872.5, asc_ords, 0, SortOrder::Ascending), asc.bisection_search_x(872.5));
    CHECK_EQ(search::hunt(872.5, asc_ords, 0, SortOrder::Ascending), asc.hunt_search_x(872.5));

    CHECK_EQ(search::sequential(872.5, dsc_ords, 0, SortOrder::Descending), dsc.sequential_search_x(872.5));
    CHECK_EQ(search::bisection(872.5, dsc_ords, 0, SortOrder::Descending), dsc.bisection_search_x(872.5));
    CHECK_EQ(search::hunt(872.5, dsc_ords, 0, SortOrder::Descending), dsc.hunt_search_x(872.5));
}

// -------------------------------------------------------------------------------------------
// Interpolation transform table (7 forward tests + 7 Rev tests = the fourteen interpolation
// tests), all at 1e-6.
// -------------------------------------------------------------------------------------------

// C# Test_Lin.
void test_lin() {
    std::vector<double> x_arr = {50, 100, 150, 200, 250};
    std::vector<double> y_arr = {100, 200, 300, 400, 500};
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Ascending, true, SortOrder::Ascending);
    double x = 75;
    double y = opd.get_y_from_x(x);
    CHECK_NEAR(y, 150.0, 1e-6);
    double x_from_y = opd.get_x_from_y(y);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_LinLog.
void test_lin_log() {
    std::vector<double> x_arr = {50, 100, 150, 200, 250};
    std::vector<double> y_arr = {100, 200, 300, 400, 500};
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Ascending, true, SortOrder::Ascending);
    double x = 75;
    double y = opd.get_y_from_x(x, Transform::None, Transform::Logarithmic);
    CHECK_NEAR(y, 141.42135623731, 1e-6);
    double x_from_y = opd.get_x_from_y(y, Transform::None, Transform::Logarithmic);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_LogLin.
void test_log_lin() {
    std::vector<double> x_arr = {50, 100, 150, 200, 250};
    std::vector<double> y_arr = {100, 200, 300, 400, 500};
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Ascending, true, SortOrder::Ascending);
    double x = 75;
    double y = opd.get_y_from_x(x, Transform::Logarithmic, Transform::None);
    CHECK_NEAR(y, 158.496250072116, 1e-6);
    double x_from_y = opd.get_x_from_y(y, Transform::Logarithmic, Transform::None);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_LogLog.
void test_log_log() {
    std::vector<double> x_arr = {50, 100, 150, 200, 250};
    std::vector<double> y_arr = {100, 200, 300, 400, 500};
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Ascending, true, SortOrder::Ascending);
    double x = 75;
    double y = opd.get_y_from_x(x, Transform::Logarithmic, Transform::Logarithmic);
    CHECK_NEAR(y, 150.0, 1e-6);
    double x_from_y = opd.get_x_from_y(y, Transform::Logarithmic, Transform::Logarithmic);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_LinZ.
void test_lin_z() {
    std::vector<double> x_arr = {0.05, 0.1, 0.15, 0.2, 0.25};
    std::vector<double> y_arr = {0.1, 0.2, 0.3, 0.4, 0.5};
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Ascending, true, SortOrder::Ascending);
    double x = 0.18;
    double y = opd.get_y_from_x(x, Transform::None, Transform::NormalZ);
    CHECK_NEAR(y, 0.358762529, 1e-6);
    double x_from_y = opd.get_x_from_y(y, Transform::None, Transform::NormalZ);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_ZLin.
void test_z_lin() {
    std::vector<double> x_arr = {0.05, 0.1, 0.15, 0.2, 0.25};
    std::vector<double> y_arr = {0.1, 0.2, 0.3, 0.4, 0.5};
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Ascending, true, SortOrder::Ascending);
    double x = 0.18;
    double y = opd.get_y_from_x(x, Transform::NormalZ, Transform::None);
    CHECK_NEAR(y, 0.362146174, 1e-6);
    double x_from_y = opd.get_x_from_y(y, Transform::NormalZ, Transform::None);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_ZZ.
void test_zz() {
    std::vector<double> x_arr = {0.05, 0.1, 0.15, 0.2, 0.25};
    std::vector<double> y_arr = {0.1, 0.2, 0.3, 0.4, 0.5};
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Ascending, true, SortOrder::Ascending);
    double x = 0.18;
    double y = opd.get_y_from_x(x, Transform::NormalZ, Transform::NormalZ);
    CHECK_NEAR(y, 0.36093855992815, 1e-6);
    double x_from_y = opd.get_x_from_y(y, Transform::NormalZ, Transform::NormalZ);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_RevLinear.
void test_rev_linear() {
    std::vector<double> x_arr = reversed({50, 100, 150, 200, 250});
    std::vector<double> y_arr = reversed({100, 200, 300, 400, 500});
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Descending, true, SortOrder::Descending);
    double x = 75;
    double y = opd.get_y_from_x(x);
    CHECK_NEAR(y, 150.0, 1e-6);
    double x_from_y = opd.get_x_from_y(y);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_RevLinLog.
void test_rev_lin_log() {
    std::vector<double> x_arr = reversed({50, 100, 150, 200, 250});
    std::vector<double> y_arr = reversed({100, 200, 300, 400, 500});
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Descending, true, SortOrder::Descending);
    double x = 75;
    double y = opd.get_y_from_x(x, Transform::None, Transform::Logarithmic);
    CHECK_NEAR(y, 141.42135623731, 1e-6);
    double x_from_y = opd.get_x_from_y(y, Transform::None, Transform::Logarithmic);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_RevLogLin.
void test_rev_log_lin() {
    std::vector<double> x_arr = reversed({50, 100, 150, 200, 250});
    std::vector<double> y_arr = reversed({100, 200, 300, 400, 500});
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Descending, true, SortOrder::Descending);
    double x = 75;
    double y = opd.get_y_from_x(x, Transform::Logarithmic, Transform::None);
    CHECK_NEAR(y, 158.496250072116, 1e-6);
    double x_from_y = opd.get_x_from_y(y, Transform::Logarithmic, Transform::None);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_RevLogLog.
void test_rev_log_log() {
    std::vector<double> x_arr = reversed({50, 100, 150, 200, 250});
    std::vector<double> y_arr = reversed({100, 200, 300, 400, 500});
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Descending, true, SortOrder::Descending);
    double x = 75;
    double y = opd.get_y_from_x(x, Transform::Logarithmic, Transform::Logarithmic);
    CHECK_NEAR(y, 150.0, 1e-6);
    double x_from_y = opd.get_x_from_y(y, Transform::Logarithmic, Transform::Logarithmic);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_RevLinZ.
void test_rev_lin_z() {
    std::vector<double> x_arr = reversed({0.05, 0.1, 0.15, 0.2, 0.25});
    std::vector<double> y_arr = reversed({0.1, 0.2, 0.3, 0.4, 0.5});
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Descending, true, SortOrder::Descending);
    double x = 0.18;
    double y = opd.get_y_from_x(x, Transform::None, Transform::NormalZ);
    CHECK_NEAR(y, 0.358762529, 1e-6);
    double x_from_y = opd.get_x_from_y(y, Transform::None, Transform::NormalZ);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_RevZLin.
void test_rev_z_lin() {
    std::vector<double> x_arr = reversed({0.05, 0.1, 0.15, 0.2, 0.25});
    std::vector<double> y_arr = reversed({0.1, 0.2, 0.3, 0.4, 0.5});
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Descending, true, SortOrder::Descending);
    double x = 0.18;
    double y = opd.get_y_from_x(x, Transform::NormalZ, Transform::None);
    CHECK_NEAR(y, 0.362146174, 1e-6);
    double x_from_y = opd.get_x_from_y(y, Transform::NormalZ, Transform::None);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_RevZZ.
void test_rev_zz() {
    std::vector<double> x_arr = reversed({0.05, 0.1, 0.15, 0.2, 0.25});
    std::vector<double> y_arr = reversed({0.1, 0.2, 0.3, 0.4, 0.5});
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Descending, true, SortOrder::Descending);
    double x = 0.18;
    double y = opd.get_y_from_x(x, Transform::NormalZ, Transform::NormalZ);
    CHECK_NEAR(y, 0.36093855992815, 1e-6);
    double x_from_y = opd.get_x_from_y(y, Transform::NormalZ, Transform::NormalZ);
    CHECK_NEAR(x_from_y, x, 1e-6);
}

// C# Test_Lin_List: the vector overloads reproduce the curve's own arrays FROM INDEX 1 (index 0
// is the clamped endpoint, deliberately skipped by the C# loop bound `i = 1`).
void test_lin_list() {
    std::vector<double> x_arr = {50, 100, 150, 200, 250};
    std::vector<double> y_arr = {100, 200, 300, 400, 500};
    OrderedPairedData opd(x_arr, y_arr, true, SortOrder::Ascending, true, SortOrder::Ascending);
    double x = 75;
    double y = opd.get_y_from_x(x);
    CHECK_NEAR(y, 150.0, 1e-6);

    auto y_vals = opd.get_y_from_x(x_arr);
    for (std::size_t i = 1; i < y_arr.size(); ++i) CHECK_NEAR(y_vals[i], y_arr[i], 1e-6);

    auto x_from_y = opd.get_x_from_y(y_vals);
    for (std::size_t i = 1; i < y_arr.size(); ++i) CHECK_NEAR(x_from_y[i], x_arr[i], 1e-6);
}

// -------------------------------------------------------------------------------------------
// Line simplification: shared five-point sin curve, same expected four points for all three
// algorithms.
// -------------------------------------------------------------------------------------------

std::vector<Ordinate> sin_curve_data() {
    return {
        Ordinate(0, 0),
        Ordinate(3.14 / 2, 1),
        Ordinate(3.14, 0),
        Ordinate(3 * 3.14 / 2, -1),
        Ordinate(2 * 3.14, 0),
    };
}

// Corehydro addition, clearly marked: the C# assertion loops below are bounded by
// `test.Count` (the RESULT length), so an implementation that returns too few points would
// pass upstream's own assertions vacuously. This helper asserts the result length FIRST
// (exactly 4, matching `valid`), then compares elementwise -- the same class of weak-assertion
// supplement P3 added for the optimizer suites.
void check_simplified(const OrderedPairedData& test, const std::vector<Ordinate>& valid) {
    CHECK_EQ(test.count(), static_cast<int>(valid.size()));
    for (std::size_t i = 0; i < valid.size() && i < static_cast<std::size_t>(test.count()); ++i) {
        CHECK_EQ(test[static_cast<int>(i)].x, valid[i].x);
        CHECK_EQ(test[static_cast<int>(i)].y, valid[i].y);
    }
}

// C# Test_DouglasPeuckerSimplify.
void test_douglas_peucker_simplify() {
    auto data = sin_curve_data();
    OrderedPairedData ordered_pair(data, true, SortOrder::Ascending, false, SortOrder::None);
    auto test = ordered_pair.douglas_peucker_simplify(0.01);
    std::vector<Ordinate> valid = {Ordinate(0, 0), Ordinate(1.57, 1), Ordinate(4.71, -1),
                                    Ordinate(6.28, 0)};
    check_simplified(test, valid);
}

// C# Test_VisvaligamWhyattSimplify.
void test_visvaligam_whyatt_simplify() {
    auto data = sin_curve_data();
    OrderedPairedData ordered_pair(data, true, SortOrder::Ascending, false, SortOrder::None);
    auto test = ordered_pair.visvaligam_whyatt_simplify(4);
    std::vector<Ordinate> valid = {Ordinate(0, 0), Ordinate(1.57, 1), Ordinate(4.71, -1),
                                    Ordinate(6.28, 0)};
    check_simplified(test, valid);
}

// Corehydro addition (P4 whole-branch-review finding M1): `num_to_keep` values that leave fewer
// than 3 ordinates to triangulate at some point in the reduction now throw std::out_of_range
// (matching the C# List<T> indexer's ArgumentOutOfRangeException) instead of reading past the end
// of the working vector. `num_to_keep = 0` and `-1` crashed both R and Python outright before this
// fix; `num_to_keep = 1` silently returned an out-of-bounds-read value (worse: no crash, wrong
// answer). `num_to_keep = 2` is the smallest value that does NOT trip the guard on this five-point
// curve (see the header's transcription note 7), so it is asserted here as the boundary that must
// still succeed.
void test_visvaligam_whyatt_simplify_out_of_range() {
    auto data = sin_curve_data();
    OrderedPairedData ordered_pair(data, true, SortOrder::Ascending, false, SortOrder::None);
    CHECK_THROWS(ordered_pair.visvaligam_whyatt_simplify(0));
    CHECK_THROWS(ordered_pair.visvaligam_whyatt_simplify(-1));
    CHECK_THROWS(ordered_pair.visvaligam_whyatt_simplify(1));
    // num_to_keep = 2 must NOT throw: the reduction reaches exactly 3 ordinates (the smallest
    // valid triangulation) on its last successful iteration and never revisits ords[2] below that.
    auto test = ordered_pair.visvaligam_whyatt_simplify(2);
    CHECK_EQ(test.count(), 2);
}

// C# Test_LangSimplify. NOTE: the C# test's own `valid` array claims four points, matching
// douglas_peucker/visvaligam_whyatt -- but LangSimplify does not force-keep the trailing point
// (see ordered_paired_data.hpp's sixth transcription finding), and this is verified DIRECTLY
// against the real C# library (`dotnet run` against upstream/Numerics @ 2a0357a): the real
// `LangSimplify(0.01, 2)` on this exact curve returns 3 points, dropping (6.28, 0). Upstream's
// own test never notices because its assertion loop is bounded by the (short) result length,
// not `valid.Count`. This port's expected value below is the VERIFIED real-library result, not
// the brief's/upstream test's claimed four points.
void test_lang_simplify() {
    auto data = sin_curve_data();
    OrderedPairedData ordered_pair(data, true, SortOrder::Ascending, false, SortOrder::None);
    auto test = ordered_pair.lang_simplify(0.01, 2);
    std::vector<Ordinate> valid = {Ordinate(0, 0), Ordinate(1.57, 1), Ordinate(4.71, -1)};
    check_simplified(test, valid);
}

// Corehydro addition (not in the C# suite -- upstream's Test_LangSimplify uses look_ahead=2,
// tolerance=0.01, neither of which trips the guard): exercises transcription note 5's guarded
// branch (look_ahead <= 1 || tolerance <= 0) and asserts THIS PORT'S chosen behavior -- a
// content-equal, independently-mutable clone, not (impossible in C++ for a value-returning
// signature) an alias of `*this`.
void test_lang_simplify_guard() {
    auto data = sin_curve_data();
    OrderedPairedData ordered_pair(data, true, SortOrder::Ascending, false, SortOrder::None);

    // tolerance <= 0 trips the guard.
    auto guarded = ordered_pair.lang_simplify(0.0, 2);
    CHECK_TRUE(guarded == ordered_pair);
    CHECK_EQ(guarded.count(), ordered_pair.count());

    // look_ahead <= 1 trips the guard.
    auto guarded2 = ordered_pair.lang_simplify(0.01, 1);
    CHECK_TRUE(guarded2 == ordered_pair);

    // Prove independence: mutating the guarded return must not mutate the original (this is
    // trivially true of C++ value semantics, but the whole point of transcription note 5 is to
    // make that choice explicit and test it rather than leave it implicit).
    int original_count = ordered_pair.count();
    guarded.add(Ordinate(99, 99));
    CHECK_EQ(ordered_pair.count(), original_count);
    CHECK_EQ(guarded.count(), original_count + 1);
}

}  // namespace

int main() {
    test_get_y();
    test_get_x();
    test_indexing();
    test_equality();
    test_trapezoidal_area();
    test_sequential();
    test_bisection();
    test_hunt();
    test_search_cross_check();
    test_lin();
    test_lin_log();
    test_log_lin();
    test_log_log();
    test_lin_z();
    test_z_lin();
    test_zz();
    test_rev_linear();
    test_rev_lin_log();
    test_rev_log_lin();
    test_rev_log_log();
    test_rev_lin_z();
    test_rev_z_lin();
    test_rev_zz();
    test_lin_list();
    test_douglas_peucker_simplify();
    test_visvaligam_whyatt_simplify();
    test_visvaligam_whyatt_simplify_out_of_range();
    test_lang_simplify();
    test_lang_simplify_guard();
    return chtest::summary("test_ordered_paired_data");
}
