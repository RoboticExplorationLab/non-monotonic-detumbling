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

tspan = (0.0, 2 * 60 * 60.0)

get_initial_state = mc_setup_get_initial_state(
    (400e3, 400e3), #h_range
    (0.0, 0.0), #e_range
    (0, pi), #i_range
    (0, 2 * pi), #Ω_range
    (0, 2 * pi), #ω_range
    (0, 2 * pi), #M_range
    (deg2rad(10), deg2rad(10)), # angular_rate_magnitude_range
)

controllers = Dict(
    "B-Cross" => (x_, t_, p_) -> bcross_control(x_, t_, p_; k=4e-6, saturate=true),
    "Lyapunov Momentum" => (x_, t_, p_) -> bmomentum_control(x_, t_, p_; k=2e3, saturate=true),
    "B-Dot Variant" => (x_, t_, p_) -> bdot_variant_autodiff(x_, t_, p_; k=4e-6, saturate=true),
    "Projection-based" => (x, t, m) -> projection_control(x, t, m; k1=10.0, k2=10.0, saturate=true),
    "Discrete Non-monotonic" => (x_, t_, p_) -> bderivative_control(x_, t_, p_; k=3e2, saturate=true, α=100),
    "Barbalat's Constrained" => (x_, t_, p_) -> bbarbalat_minVd(x_, t_, p_; k=1e2, saturate=true),
)

Ntrials = 100


mc_results = monte_carlo_orbit_attitude(get_initial_state, controllers, Ntrials, params, tspan; integrator_dt=0.1, controller_dt=0.0)


datafilename = "mc_orbit_varied_all.jld2"
datapath = joinpath(@__DIR__, "..", "data", datafilename)
print("Saving data to $datapath")
save(datapath, Dict("mc_results" => mc_results, "params" => toDict(params)))
