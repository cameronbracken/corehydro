// ported from: Numerics/Mathematics/ODE Solvers/RungeKutta.cs @ 2a0357a
//
// The Runge-Kutta family of ODE solvers: second- and fourth-order fixed-step methods (each
// returning the full solution array over an equally-spaced grid, plus a single-step overload of
// the fourth-order method), and the two adaptive-step-size methods, Runge-Kutta-Fehlberg and
// Runge-Kutta-Cash-Karp. Free functions, `rootfinding/brent.hpp` style, over `f(t, y)`.
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace corehydro::numerics::math::ode {

// Second-order (midpoint/Heun-style) Runge-Kutta over `time_steps` equally-spaced points from
// `start_time` to `end_time`, inclusive of both ends. Returns a vector of length `time_steps`
// (NOT `time_steps + 1` -- the ported C# allocates `new double[timeSteps]`).
inline std::vector<double> second_order(const std::function<double(double, double)>& f,
                                        double initial_value, double start_time, double end_time,
                                        int time_steps) {
    double dt = (end_time - start_time) / (time_steps - 1);
    double t = start_time;
    std::vector<double> y(static_cast<std::size_t>(time_steps));
    double y0 = initial_value;
    y[0] = y0;
    for (int i = 1; i < time_steps; ++i) {
        double k1 = f(t, y0);
        double k2 = f(t + dt, y0 + k1 * dt);
        y[static_cast<std::size_t>(i)] = y0 + dt * 0.5 * (k1 + k2);
        t += dt;
        y0 = y[static_cast<std::size_t>(i)];
    }
    return y;
}

// The fourth-order Runge-Kutta method over `time_steps` equally-spaced points, the same shape as
// `second_order` above.
inline std::vector<double> fourth_order(const std::function<double(double, double)>& f,
                                        double initial_value, double start_time, double end_time,
                                        int time_steps) {
    double dt = (end_time - start_time) / (time_steps - 1);
    double t = start_time;
    std::vector<double> y(static_cast<std::size_t>(time_steps));
    double y0 = initial_value;
    y[0] = y0;
    for (int i = 1; i < time_steps; ++i) {
        double k1 = f(t, y0);
        double k2 = f(t + dt / 2.0, y0 + k1 * dt / 2.0);
        double k3 = f(t + dt / 2.0, y0 + k2 * dt / 2.0);
        double k4 = f(t + dt, y0 + k3 * dt);
        y[static_cast<std::size_t>(i)] = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += dt;
        y0 = y[static_cast<std::size_t>(i)];
    }
    return y;
}

// The fourth-order method's single-step overload: the ODE solved at `start_time + dt`. A caller
// wanting a trajectory calls this in a loop, feeding each result back in as the next
// `initial_value`/`start_time` -- exactly as Test_RungeKutta.cs's own Test_4thRK_2 does.
inline double fourth_order_step(const std::function<double(double, double)>& f,
                                double initial_value, double start_time, double dt) {
    double t = start_time;
    double y = initial_value;
    double k1 = f(t, y);
    double k2 = f(t + dt / 2.0, y + k1 * dt / 2.0);
    double k3 = f(t + dt / 2.0, y + k2 * dt / 2.0);
    double k4 = f(t + dt, y + k3 * dt);
    return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// The ported defaults, named rather than left as a bare literal in the signature -- the same
// convention brent.hpp's kDefaultTolerance follows, and for the same reason: callback/math.hpp's
// ode_solve arm is the one caller that must not restate this value.
inline constexpr double kDefaultOdeTolerance = 1E-3;

// The adaptive Runge-Kutta-Fehlberg method: advances from `start_time` to `start_time + dt` with
// an internal step size bounded above by `dt` and below by `dt_min`, halving the step on
// rejection and doubling it once the error estimate falls comfortably inside `tolerance`.
inline double fehlberg(const std::function<double(double, double)>& f, double initial_value,
                       double start_time, double dt, double dt_min,
                       double tolerance = kDefaultOdeTolerance) {
    double t = start_time;
    double tf = start_time + dt;
    double y = initial_value;
    double h = dt;
    double min_tolerance = 1E-2 * tolerance;

    while (t < tf) {
        if (h < dt_min) h = dt_min;
        if (h > dt) h = dt;
        if (t + h > tf) h = tf - t;

        double k1 = h * f(t, y);
        double k2 = h * f(t + 1.0 / 4.0 * h, y + 1.0 / 4.0 * k1);
        double k3 = h * f(t + 3.0 / 8.0 * h, y + 3.0 / 32.0 * k1 + 9.0 / 32.0 * k2);
        double k4 = h * f(t + 12.0 / 13.0 * h, y + 1932.0 / 2197.0 * k1 - 7200.0 / 2197.0 * k2 +
                                                   7296.0 / 2197.0 * k3);
        double k5 = h * f(t + 1.0 / 1.0 * h, y + 439.0 / 216.0 * k1 - 8.0 * k2 +
                                                 3680.0 / 513.0 * k3 - 845.0 / 4104.0 * k4);
        double k6 = h * f(t + 1.0 / 2.0 * h, y - 8.0 / 27.0 * k1 + 2.0 * k2 -
                                                 3544.0 / 2565.0 * k3 + 1859.0 / 4104.0 * k4 -
                                                 11.0 / 40.0 * k5);
        double rk4 = y + 25.0 / 216.0 * k1 + 1408.0 / 2565.0 * k3 + 2197.0 / 4104.0 * k4 -
                    1.0 / 5.0 * k6;
        double rk5 = y + 16.0 / 135.0 * k1 + 6656.0 / 12825.0 * k3 + 28561.0 / 56430.0 * k4 -
                    9.0 / 50.0 * k5 + 2.0 / 55.0 * k6;
        double error = std::max(min_tolerance, std::fabs(rk4 - rk5));

        if (error <= tolerance || h <= dt_min) {
            t += h;
            y = rk5;
            if (error < min_tolerance) h *= 2.0;
        } else {
            h /= 2.0;
        }
    }

    return y;
}

// The adaptive Runge-Kutta-Cash-Karp method, the same step-control loop as `fehlberg` above over
// the Cash-Karp coefficients.
inline double cash_karp(const std::function<double(double, double)>& f, double initial_value,
                        double start_time, double dt, double dt_min,
                        double tolerance = kDefaultOdeTolerance) {
    double t = start_time;
    double tf = start_time + dt;
    double y = initial_value;
    double h = dt;
    double min_tolerance = 1E-2 * tolerance;

    while (t < tf) {
        if (h < dt_min) h = dt_min;
        if (h > dt) h = dt;
        if (t + h > tf) h = tf - t;

        double k1 = h * f(t, y);
        double k2 = h * f(t + 1.0 / 5.0 * h, y + 1.0 / 5.0 * k1);
        double k3 = h * f(t + 3.0 / 10.0 * h, y + 3.0 / 40.0 * k1 + 9.0 / 40.0 * k2);
        double k4 = h * f(t + 3.0 / 5.0 * h,
                          y + 3.0 / 10.0 * k1 - 9.0 / 10.0 * k2 + 6.0 / 5.0 * k3);
        double k5 = h * f(t + 1.0 / 1.0 * h, y - 11.0 / 54.0 * k1 + 5.0 / 2.0 * k2 -
                                                 70.0 / 27.0 * k3 + 35.0 / 27.0 * k4);
        double k6 =
            h * f(t + 7.0 / 8.0 * h, y + 1631.0 / 55296.0 * k1 + 175.0 / 512.0 * k2 +
                                         575.0 / 13824.0 * k3 + 44275.0 / 110592.0 * k4 +
                                         253.0 / 4096.0 * k5);
        double rk4 =
            y + 37.0 / 378.0 * k1 + 250.0 / 621.0 * k3 + 125.0 / 594.0 * k4 + 512.0 / 1771.0 * k6;
        double rk5 = y + 2825.0 / 27648.0 * k1 + 18575.0 / 48384.0 * k3 +
                    13525.0 / 55296.0 * k4 + 277.0 / 14336.0 * k5 + 1.0 / 4.0 * k6;
        double error = std::max(min_tolerance, std::fabs(rk4 - rk5));

        if (error <= tolerance || h <= dt_min) {
            t += h;
            y = rk5;
            if (error < min_tolerance) h *= 2.0;
        } else {
            h /= 2.0;
        }
    }

    return y;
}

}  // namespace corehydro::numerics::math::ode
