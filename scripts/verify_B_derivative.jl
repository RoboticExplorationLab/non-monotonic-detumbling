using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2
using Random
using Plots
plotly()

include("../src/satellite_simulator.jl")
include("../src/detumble_controller.jl")
include("../src/satellite_models.jl")

let
    params_no_noise = OrbitDynamicsParameters(py4_model_no_noise;
        distance_scale=1.0,
        time_scale=1.0,
        angular_rate_scale=1.0,
        control_scale=1,
        control_type=:dipole,
        magnetic_model=:IGRF13,
        add_solar_radiation_pressure=false,
        add_sun_thirdbody=false,
        add_moon_thirdbody=false)

    params_noisy = OrbitDynamicsParameters(py4_model;
        distance_scale=1.0,
        time_scale=1.0,
        angular_rate_scale=1.0,
        control_scale=1,
        control_type=:dipole,
        magnetic_model=:IGRF13,
        add_solar_radiation_pressure=false,
        add_sun_thirdbody=false,
        add_moon_thirdbody=false)

    tspan = (0.0, 2 * 60 * 60.0)
    integrator_dt = 0.1
    controller_dt = 0.0


    x0 = [
        -6.563459794497504e6
        -504695.1974266482
        -1.6153668267879453e6
        484.36185528391604
        -7641.740424953377
        419.5085091962227
        0.17544901540778812
        0.45892815737289877
        0.8162933673410933
        0.3037560333519198
        0.026154050372741044
        0.0760906576674854
        0.5173797978978956
    ]

    dt_orbit = 1.0
    orbit_time = floor(tspan[2] - tspan[1])
    Nt_orbit = Int(orbit_time / dt_orbit)
    orbital_states = propagate_orbit(x0[1:6], dt_orbit, Nt_orbit)
    B_vectors = compute_B_vectors(params_noisy, orbital_states, dt_orbit)
    B_interp = interpolate_B_vectors(B_vectors, 0:dt_orbit:orbit_time)


    Ntimesteps = Int(ceil((tspan[2] - tspan[1]) / integrator_dt))
    B_hist_true = zeros(3, Ntimesteps)
    Bdot_hist_true = zeros(3, Ntimesteps)
    B_hist_filtered = zeros(3, Ntimesteps)
    Bdot_hist_filtered = zeros(3, Ntimesteps)
    Bdot_hist_stencil = zeros(3, Ntimesteps)
    B2_hist = zeros(3, Ntimesteps)
    B2_hist_true = zeros(3, Ntimesteps)

    B_control_buffer = [zeros(3) for i = 1:4]
    B_stencil_buffer = [zeros(3) for i = 1:4]
    B_filtered_buffer = [zeros(3) for i = 1:4]

    hist_index = 1
    function bderivative_with_B_logging(x_, t_, p_)
        B_body = measure_magnetic_B_vector_body(x_, t_, p_) # magnetometer measurement
        B_hist_true[:, hist_index] .= B_body
        Bdot_hist_true[:, hist_index] .= measure_magnetic_B_vector_body_dot(x_, t_, p_) # derivative of magnetometer measurement
        _, Bdot_stencil = update_bdot_estimate(B_stencil_buffer, B_body, integrator_dt)
        Bdot_hist_stencil[:, hist_index] .= Bdot_stencil
        B_filtered, Bdot_filtered = five_sample_polynomial_filter(B_filtered_buffer, B_body, integrator_dt)
        B_hist_filtered[:, hist_index] .= B_filtered
        Bdot_hist_filtered[:, hist_index] .= Bdot_filtered

        tderivative = 10 * 60
        ω_body_wrt_inertial_in_body = x_[11:13]
        B1_dot_orbit_wrt_inertial_in_body = Bdot_hist_true[:, hist_index] - hat(B_body) * ω_body_wrt_inertial_in_body
        # B1_dot_orbit_wrt_inertial_in_body = Bdot_filtered - hat(B_filtered) * ω_body_wrt_inertial_in_body
        # B2_body = B_filtered + tderivative * B1_dot_orbit_wrt_inertial_in_body
        B2_body = B_body + tderivative * B1_dot_orbit_wrt_inertial_in_body
        B2_hist[:, hist_index] .= Q(x_[7:10]) * B2_body
        B2_hist_true[:, hist_index] = B_interp(t_ + tderivative)

        hist_index += 1
        return bderivative_control(x_, t_, p_; k=3e2, saturate=true, α=100.0, tderivative=tderivative, Bhist=B_control_buffer, time_step=integrator_dt)
    end

    xhist, uhist, thist = simulate_satellite_orbit_attitude_rk4(x0, params_no_noise, tspan; integrator_dt=integrator_dt, controller=bderivative_with_B_logging, controller_dt=controller_dt)

    # max_samples = 1000
    downsample = 1:1000 # get_downsample(length(thist), max_samples)
    ω = xhist[11:13, downsample]
    J = params_no_noise.satellite_model.inertia
    h = J * ω
    h̄ = dropdims(sqrt.(sum(h .* h, dims=1)); dims=1)
    tplot = thist[downsample]
    display(plot(tplot, h̄, title="Momentum Magnitude", xlabel="Time (seconds)", ylabel="||h|| (Nms)"))

    Bx_plot = plot(tplot, B_hist_true[1, downsample], label="True", xlabel="Time (seconds)", ylabel="B_x (T)")
    By_plot = plot(tplot, B_hist_true[2, downsample], label="", xlabel="Time (seconds)", ylabel="B_y (T)")
    Bz_plot = plot(tplot, B_hist_true[3, downsample], label="", xlabel="Time (seconds)", ylabel="B_z (T)")
    plot!(Bx_plot, tplot, B_hist_filtered[1, downsample], label="Filtered", xlabel="Time (seconds)", ylabel="B_x (T)")
    plot!(By_plot, tplot, B_hist_filtered[2, downsample], label="", xlabel="Time (seconds)", ylabel="B_y (T)")
    plot!(Bz_plot, tplot, B_hist_filtered[3, downsample], label="", xlabel="Time (seconds)", ylabel="B_z (T)")

    display(plot(Bx_plot, By_plot, Bz_plot, layout=(3, 1), title="B-vector comparison"))

    #  plot(tplot, Bdot_hist_true[1, downsample], label="True", xlabel="Time (seconds)", ylabel="Ḃ_x (T/s)")
    #  plot(tplot, Bdot_hist_true[2, downsample], label="", xlabel="Time (seconds)", ylabel="Ḃ_y (T/s)")
    #  plot(tplot, Bdot_hist_true[3, downsample], label="", xlabel="Time (seconds)", ylabel="Ḃ_z (T/s)")
    Bx_dot_plot = plot(tplot, (Bdot_hist_true[1, downsample] .- Bdot_hist_filtered[1, downsample]), label="Filtered", xlabel="Time (seconds)", ylabel="Ḃ_x (T/s)")
    By_dot_plot = plot(tplot, (Bdot_hist_true[2, downsample] .- Bdot_hist_filtered[2, downsample]), label="", xlabel="Time (seconds)", ylabel="Ḃ_y (T/s)")
    Bz_dot_plot = plot(tplot, (Bdot_hist_true[3, downsample] .- Bdot_hist_filtered[3, downsample]), label="", xlabel="Time (seconds)", ylabel="Ḃ_z (T/s)")
    # plot!(Bx_dot_plot, tplot, Bdot_hist_true[1, downsample] .- Bdot_hist_stencil[1, downsample], label="Stencil", xlabel="Time (seconds)", ylabel="Ḃ_x (T/s)")
    # plot!(By_dot_plot, tplot, Bdot_hist_true[2, downsample] .- Bdot_hist_stencil[2, downsample], label="", xlabel="Time (seconds)", ylabel="Ḃ_y (T/s)")
    # plot!(Bz_dot_plot, tplot, Bdot_hist_true[3, downsample] .- Bdot_hist_stencil[3, downsample], label="", xlabel="Time (seconds)", ylabel="Ḃ_z (T/s)")
    display(plot(Bx_dot_plot, By_dot_plot, Bz_dot_plot, layout=(3, 1), plot_title="Bdot-vector  % Error"))

    B2_x_plot = plot(tplot, (B2_hist_true[1, downsample] .- B2_hist[1, downsample]) ./ B2_hist_true[1, downsample], label="", xlabel="Time (seconds)", ylabel="B2_x (T)")
    B2_y_plot = plot(tplot, (B2_hist_true[2, downsample] .- B2_hist[2, downsample]) ./ B2_hist_true[2, downsample], label="", xlabel="Time (seconds)", ylabel="B2_y (T)")
    B2_z_plot = plot(tplot, (B2_hist_true[3, downsample] .- B2_hist[3, downsample]) ./ B2_hist_true[3, downsample], label="", xlabel="Time (seconds)", ylabel="B2_z (T)")
    display(plot(B2_x_plot, B2_y_plot, B2_z_plot, layout=(3, 1), plot_title="B2 % Error in Inertial"))
end