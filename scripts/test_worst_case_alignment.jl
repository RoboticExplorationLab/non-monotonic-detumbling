using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
import SatelliteDynamics
using Random
using LinearAlgebra
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

x_osc_0 = [400e3 + SatelliteDynamics.R_EARTH, 0.0, deg2rad(50), deg2rad(-1.0), 0.0, 0.0] # a, e, i, Ω, ω, M
# x_osc_0 = [525e3 + SatelliteDynamics.R_EARTH, 0.0001, deg2rad(97.6), deg2rad(-1.0), 0.0, 45.0] # a, e, i, Ω, ω, M
q0 = [1.0, 0.0, 0.0, 0.0]
ω0 = [0.0, 0.0, 0.0]


for jj = 1:10
    x_osc_jj = x_osc_0 .+ vcat([0, 0], 2 * pi * rand(4))
    x0 = state_from_osc(x_osc_jj, q0, ω0)
    x0 = h_B_aligned_initial_conditions(x0, deg2rad(50), params)
    BB = magnetic_B_vector_body(x0, 0.0, params)

    ν = eigen(params.satellite_model.inertia).vectors[:, 3] # maximum principle axis
    if norm(cross(BB, ν)) > 1e-12
        @show BB
        @show ν
        @infiltrate
    else
        println("$jj: Passed")
    end
end

# @show xnew
# @show BB
# @infiltrate