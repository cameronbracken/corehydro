// ported from: Numerics/Mathematics/Integration/AdaptiveGaussKronrod2D.cs @ 7e8e8d1
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/integration/support/integrator.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::integration {

class AdaptiveGaussKronrod2D : public Integrator {
   public:
    using Recorder = std::function<void(double, double, double, double)>;

    AdaptiveGaussKronrod2D(std::function<double(double, double)> function,
                           double min_x, double max_x, double min_y, double max_y)
        : function_(std::move(function)), ax_(min_x), bx_(max_x), ay_(min_y), by_(max_y) {
        if (!function_) throw std::invalid_argument("The function cannot be null.");
        if (!is_finite(min_x) || !is_finite(max_x) || !is_finite(min_y) || !is_finite(max_y))
            throw std::out_of_range("The integration bounds must be finite.");
        if (max_x <= min_x || max_y <= min_y)
            throw std::out_of_range("Each maximum bound must exceed its minimum bound.");
    }

    int min_depth = 0;
    int max_depth = 100;
    Recorder recorder;

    double min_x() const { return ax_; }
    double max_x() const { return bx_; }
    double min_y() const { return ay_; }
    double max_y() const { return by_; }
    double standard_error() const { return standard_error_; }

    void integrate() override {
        standard_error_ = 0.0;
        clear_results();
        validate();
        if (min_depth < 0) throw std::out_of_range("The minimum depth cannot be negative.");
        if (max_depth < min_depth)
            throw std::out_of_range("The maximum depth cannot be less than the minimum depth.");
        const Recorder record = recorder;
        const bool capture = static_cast<bool>(record);
        try {
            std::vector<Region> regions;
            regions.push_back(evaluate(ax_, bx_, ay_, by_, 0, capture));
            std::vector<HeapEntry> heap;
            push(heap, {0, regions[0].error});
            double result_sum = regions[0].kronrod;
            double error_sum = regions[0].error;

            for (std::size_t i = 0; i < regions.size(); ++i) {
                if (regions[i].split || regions[i].frozen || regions[i].depth >= min_depth)
                    continue;
                if (function_evaluations_ >= max_function_evaluations) break;
                split_region(regions, heap, i, capture, result_sum, error_sum);
            }

            IntegrationStatus final_status = IntegrationStatus::Success;
            while (true) {
                double tolerance =
                    std::max(absolute_tolerance, relative_tolerance * std::abs(result_sum));
                if (function_evaluations_ >= min_function_evaluations && error_sum <= tolerance) {
                    final_status = IntegrationStatus::Success;
                    break;
                }
                if (function_evaluations_ >= max_function_evaluations) {
                    final_status = IntegrationStatus::MaximumFunctionEvaluationsReached;
                    break;
                }
                if (heap.empty()) {
                    final_status = IntegrationStatus::MaximumIterationsReached;
                    break;
                }
                HeapEntry entry = pop(heap);
                Region& worst = regions[entry.index];
                if (worst.split || worst.frozen) continue;
                if (worst.depth >= max_depth) {
                    worst.frozen = true;
                    continue;
                }
                split_region(regions, heap, entry.index, capture, result_sum, error_sum);
            }

            result_ = 0.0;
            standard_error_ = 0.0;
            for (const Region& region : regions) {
                if (region.split) continue;
                result_ += region.kronrod;
                standard_error_ += region.error;
            }
            status_ = final_status;
            if (record) flush_recorder(regions, record);
        } catch (...) {
            status_ = IntegrationStatus::Failure;
            if (report_failure) throw;
        }
    }

   private:
    struct Region {
        double ax = 0, bx = 0, ay = 0, by = 0;
        double kronrod = 0, error = 0, error_x = 0, error_y = 0;
        int depth = 0;
        bool split = false, frozen = false;
        std::vector<double> x_nodes, y_nodes, values;
    };
    struct HeapEntry { std::size_t index; double error; };

    std::function<double(double, double)> function_;
    double ax_, bx_, ay_, by_;
    double standard_error_ = 0.0;

    static constexpr double xk_[11] = {
        .995657163025808080735527280689003, .973906528517171720077964012084452,
        .930157491355708226001207180059508, .865063366688984510732096688423493,
        .780817726586416897063717578345042, .679409568299024406234327365114874,
        .562757134668604683339000099272694, .433395394129247190799265943165784,
        .294392862701460198131126603103866, .148874338981631210884826001129720, 0.0};
    static constexpr double wk_[11] = {
        .011694638867371874278064396062192, .032558162307964727478818972459390,
        .054755896574351996031381300244580, .075039674810919952767043140916190,
        .093125454583697605535065083366, .109387158802297641899210590325805,
        .123491976262065851077958109831074, .134709217311473325928054001771707,
        .142775938577060080797094273138717, .147739104901338491374841515972068,
        .149445554002916905664936468389821};
    static constexpr double wg_[5] = {
        .066671344308688137593568809893332, .149451349150580593145776339657697,
        .219086362515982043995534934228163, .269266719309996355091226921569469,
        .295524224714752870173892994651338};

    static std::vector<double> nodes() {
        std::vector<double> value(21);
        for (int i = 0; i < 10; ++i) { value[i] = -xk_[i]; value[20 - i] = xk_[i]; }
        return value;
    }
    static std::vector<double> weights(bool gauss) {
        std::vector<double> value(21);
        for (int p = 0; p < 21; ++p) {
            int i = p <= 10 ? p : 20 - p;
            value[p] = gauss ? (i % 2 == 1 ? wg_[i / 2] : 0.0) : wk_[i];
        }
        return value;
    }

    Region evaluate(double ax, double bx, double ay, double by, int depth, bool capture) {
        static const std::vector<double> n = nodes();
        static const std::vector<double> wk = weights(false);
        static const std::vector<double> wg = weights(true);
        Region region;
        region.ax = ax; region.bx = bx; region.ay = ay; region.by = by; region.depth = depth;
        const double cx = .5 * (ax + bx), hx = .5 * (bx - ax);
        const double cy = .5 * (ay + by), hy = .5 * (by - ay);
        std::vector<double> xs(21), ys(21);
        for (int i = 0; i < 21; ++i) { xs[i] = cx + hx * n[i]; ys[i] = cy + hy * n[i]; }
        if (capture) region.values.resize(441);
        double kk = 0, gg = 0, gxky = 0, kxgy = 0;
        for (int i = 0; i < 21; ++i) {
            double row_k = 0, row_g = 0;
            for (int j = 0; j < 21; ++j) {
                double f = function_(xs[i], ys[j]);
                if (capture) region.values[static_cast<std::size_t>(i * 21 + j)] = f;
                row_k += wk[j] * f;
                row_g += wg[j] * f;
            }
            kk += wk[i] * row_k;
            kxgy += wk[i] * row_g;
            gxky += wg[i] * row_k;
            gg += wg[i] * row_g;
        }
        function_evaluations_ += 441;
        double scale = hx * hy;
        region.kronrod = kk * scale;
        region.error = std::abs(region.kronrod - gg * scale);
        region.error_x = std::abs(region.kronrod - gxky * scale);
        region.error_y = std::abs(region.kronrod - kxgy * scale);
        if (capture) { region.x_nodes = std::move(xs); region.y_nodes = std::move(ys); }
        return region;
    }

    void split_region(std::vector<Region>& regions, std::vector<HeapEntry>& heap,
                      std::size_t index, bool capture, double& result_sum, double& error_sum) {
        Region& current = regions[index];
        bool split_x = current.error_x >= current.error_y;
        bool x_narrow = std::abs(current.bx - current.ax) <= kDoubleMachineEpsilon;
        bool y_narrow = std::abs(current.by - current.ay) <= kDoubleMachineEpsilon;
        if (split_x && x_narrow) split_x = false;
        if (!split_x && y_narrow) {
            if (x_narrow) { current.frozen = true; return; }
            split_x = true;
        }
        Region left, right;
        if (split_x) {
            double mid = .5 * (current.ax + current.bx);
            if (mid <= current.ax || mid >= current.bx) { current.frozen = true; return; }
            left = evaluate(current.ax, mid, current.ay, current.by, current.depth + 1, capture);
            right = evaluate(mid, current.bx, current.ay, current.by, current.depth + 1, capture);
        } else {
            double mid = .5 * (current.ay + current.by);
            if (mid <= current.ay || mid >= current.by) { current.frozen = true; return; }
            left = evaluate(current.ax, current.bx, current.ay, mid, current.depth + 1, capture);
            right = evaluate(current.ax, current.bx, mid, current.by, current.depth + 1, capture);
        }
        const double old_k = current.kronrod, old_e = current.error;
        current.split = true;
        current.x_nodes.clear(); current.y_nodes.clear(); current.values.clear();
        result_sum += left.kronrod + right.kronrod - old_k;
        error_sum += left.error + right.error - old_e;
        ++iterations_;
        regions.push_back(std::move(left)); push(heap, {regions.size() - 1, regions.back().error});
        regions.push_back(std::move(right)); push(heap, {regions.size() - 1, regions.back().error});
    }

    static void push(std::vector<HeapEntry>& heap, HeapEntry entry) {
        heap.push_back(entry);
        std::size_t child = heap.size() - 1;
        while (child > 0) {
            std::size_t parent = (child - 1) / 2;
            if (heap[parent].error >= heap[child].error) break;
            std::swap(heap[parent], heap[child]); child = parent;
        }
    }
    static HeapEntry pop(std::vector<HeapEntry>& heap) {
        HeapEntry top = heap.front();
        heap.front() = heap.back(); heap.pop_back();
        std::size_t parent = 0;
        while (!heap.empty()) {
            std::size_t left = 2 * parent + 1;
            if (left >= heap.size()) break;
            std::size_t right = left + 1;
            std::size_t larger = right < heap.size() && heap[right].error > heap[left].error
                                     ? right : left;
            if (heap[parent].error >= heap[larger].error) break;
            std::swap(heap[parent], heap[larger]); parent = larger;
        }
        return top;
    }
    static void flush_recorder(const std::vector<Region>& regions, const Recorder& recorder) {
        static const std::vector<double> wk = weights(false);
        for (const Region& region : regions) {
            if (region.split || region.values.empty()) continue;
            double scale = .25 * (region.bx - region.ax) * (region.by - region.ay);
            for (int i = 0; i < 21; ++i)
                for (int j = 0; j < 21; ++j)
                    recorder(region.x_nodes[i], region.y_nodes[j], wk[i] * wk[j] * scale,
                             region.values[static_cast<std::size_t>(i * 21 + j)]);
        }
    }
};

}  // namespace corehydro::numerics::math::integration
