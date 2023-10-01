using Pkg
using JLD2
Pkg.activate(joinpath(@__DIR__, ".."))
using Random
Random.seed!(0)

include("../src/satellite_simulator.jl")
include("../src/detumble_controller.jl")
include("../src/satellite_models.jl")

params = OrbitDynamicsParameters(py4_model_diagonal;
    distance_scale=1.0,
    time_scale=1.0,
    angular_rate_scale=1.0,
    control_scale=1,
    control_type=:dipole,
    magnetic_model=:IGRF13,
    add_solar_radiation_pressure=true,
    add_sun_thirdbody=true,
    add_moon_thirdbody=true)

tspan = (0.0, 4 * 60 * 60.0)

x_osc_0 = [400e3 + SatelliteDynamics.R_EARTH, 0.0, deg2rad(50), deg2rad(-1.0), 0.0, 0.0] # a, e, i, Ω, ω, M
# x_osc_0 = [525e3 + SatelliteDynamics.R_EARTH, 0.0001, deg2rad(97.6), deg2rad(-1.0), 0.0, 45.0] # a, e, i, Ω, ω, M
q0 = [1.0, 0.0, 0.0, 0.0]
ω0 = [0.0, 0.0, 0.0]
x0 = state_from_osc(x_osc_0, q0, ω0)
x0 = h_B_aligned_initial_conditions(x0, deg2rad(50), params)

integrator_dt = 0.1
controller_dt = 0.0

sweep_range = -4:1:5 #[-1.0, 0.0, 1.0]

controllers = Dict(
    "B-Cross" => Dict(
        "controller" => (x_, t_, p_, k_) -> bcross_control(x_, t_, p_; k=k_, saturate=true),
        "gains" => 4e-6 * (10.0 .^ sweep_range)
    ),
    "Lyapunov Momentum" => Dict(
        "controller" => (x_, t_, p_, k_) -> bmomentum_control(x_, t_, p_; k=k_, saturate=true),
        "gains" => 2e3 * (10.0 .^ sweep_range)
    ),
    "B-Dot Variant" => Dict(
        "controller" => (x_, t_, p_, k_) -> bdot_variant_autodiff(x_, t_, p_; k=k_, saturate=true),
        "gains" => 4e-6 * (10.0 .^ sweep_range)
    ),
    "Projection-based" => Dict(
        "controller" => (x, t, m, k_) -> projection_control(x, t, m; k1=k_, k2=10.0, saturate=true),
        "gains" => 10.0 * (10.0 .^ sweep_range)
    ),
    "Discrete Non-monotonic" => Dict(
        "controller" => (x_, t_, p_, k_) -> bderivative_control(x_, t_, p_; k=k_, saturate=true, α=100),
        "gains" => 3e2 * (10.0 .^ sweep_range)
    ),
    "Barbalat's Constrained" => Dict(
        "controller" => (x_, t_, p_, k_) -> bbarbalat_minVd(x_, t_, p_; k=k_, saturate=true),
        "gains" => 1e2 * (10.0 .^ sweep_range)
    ),
)

Ntimesteps = Int(ceil((tspan[2] - tspan[1]) / integrator_dt))
Nsteps = length(sweep_range)
Ncontrollers = length(keys(controllers))
gs_results = Dict(
    key => Dict(
        "X" => zeros(Nsteps, 13, Ntimesteps),
        "U" => zeros(Nsteps, 3, Ntimesteps),
        "T" => zeros(Nsteps, 1, Ntimesteps),
        "gains" => controllers[key]["gains"],
    ) for (key, _) in controllers)

controller_names = collect(keys(controllers))
Threads.@threads for i = eachindex(controller_names)
    controller_name = controller_names[i]
    println("Thread $(Threads.threadid()), $controller_name")
    gains = controllers[controller_name]["gains"]
    controller = controllers[controller_name]["controller"]
    for gain_idx = eachindex(gains)
        gain = gains[gain_idx]
        simulate_satellite_orbit_attitude_rk4
        xhist, uhist, thist = simulate_satellite_orbit_attitude_rk4(x0, params, tspan; integrator_dt=integrator_dt, controller=(x_, t_, p_) -> controller(x_, t_, p_, gain), controller_dt=controller_dt)
        gs_results[controller_name]["X"][gain_idx, :, :] .= xhist
        gs_results[controller_name]["U"][gain_idx, :, :] .= uhist
        gs_results[controller_name]["T"][gain_idx, 1, :] .= thist
    end
end


datafilename = "gain_sweep_all.jld2"
datapath = joinpath(@__DIR__, "..", "data", datafilename)
print("Saving data to $datapath")
save(datapath, Dict("gs_results" => gs_results, "params" => toDict(params)))

