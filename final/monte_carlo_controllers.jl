using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2
using Random

include("../src/satellite_simulator.jl")
include("../src/detumble_controller.jl")
include("../src/satellite_models.jl")

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

get_initial_state = mc_setup_get_initial_state(
    (400e3, 400e3), #h_range
    (0.0, 0.0), #e_range
    (deg2rad(20), deg2rad(160)), #i_range
    (0, 2 * pi), #Ω_range
    (0, 2 * pi), #ω_range
    (0, 2 * pi), #M_range
    (deg2rad(30), deg2rad(30)), # angular_rate_magnitude_range
)

controllers = Dict(
    "B-cross" => (x_, t_, p_, B_) -> bcross_control(x_, t_, p_; k=4e-5, saturate=true),
    "Lyapunov Momentum" => (x_, t_, p_, B_) -> bmomentum_control(x_, t_, p_; k=2e3, saturate=true),
    "B-dot Variant" => (x_, t_, p_, B_) -> bdot_variant(x_, t_, p_; k=0.4, saturate=true, Bhist=B_, time_step=integrator_dt),
    "B-dot" => (x_, t_, p_, B_) -> bdot_control(x_, t_, p_; k=1.0, saturate=true, Bhist=B_, time_step=integrator_dt),
    "Projection-based" => (x_, t_, m_, B_) -> projection_control(x_, t_, m_; k1=5e-2, k2=4.0, saturate=true),
    "Discrete Non-monotonic" => (x_, t_, p_, B_) -> bderivative_control(x_, t_, p_; k=3e3, saturate=true, α=100.0, tderivative=10 * 60, Bhist=B_, time_step=integrator_dt),
)

Ntrials = 100


Random.seed!(0)
mc_results_no_noise = monte_carlo_orbit_attitude(get_initial_state, controllers, Ntrials, params_no_noise, tspan; integrator_dt=integrator_dt, controller_dt=controller_dt)

datafilename = "mc_orbit_varied_no_noise_30deg_s.jld2"
datapath = joinpath(@__DIR__, "..", "data", datafilename)
print("Saving data to $datapath")
save(datapath, Dict("mc_results" => mc_results_no_noise, "params" => toDict(params_no_noise)))

Random.seed!(0)
mc_results_noisy = monte_carlo_orbit_attitude(get_initial_state, controllers, Ntrials, params_noisy, tspan; integrator_dt=integrator_dt, controller_dt=controller_dt)

datafilename = "mc_orbit_varied_noisy_30deg_s.jld2"
datapath = joinpath(@__DIR__, "..", "data", datafilename)
print("Saving data to $datapath")
save(datapath, Dict("mc_results" => mc_results_noisy, "params" => toDict(params_noisy)))