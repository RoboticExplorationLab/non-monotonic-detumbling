using Infiltrator
import SatelliteDynamics
using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using Plots
using ProgressLogging
plotly()

include("quaternions.jl")
include("satellite_models.jl")
include("magnetic_field.jl")

struct OrbitDynamicsParameters
    satellite_model::SatelliteModel
    # scales
    distance_scale::Real
    time_scale::Real
    control_scale::Real
    angular_rate_scale::Real
    # gravity fidelity
    n_gravity::Int
    m_gravity::Int
    # time
    start_epoch::SatelliteDynamics.Epoch # the epoch at which the dynamics integration begins
    # flags
    control_type::Symbol # one of :thrust :drag_ratio :torque or :dipole
    magnetic_model::Symbol
    add_solar_radiation_pressure::Bool
    add_sun_thirdbody::Bool
    add_moon_thirdbody::Bool
    add_gravity_gradient_torque::Bool
end

function OrbitDynamicsParameters(
    satellite_model::SatelliteModel;
    distance_scale::Real=1.0,
    time_scale::Real=1.0,
    control_scale::Real=1.0,
    angular_rate_scale::Real=1.0,
    n_gravity::Int=2,
    m_gravity::Int=0,
    start_epoch::SatelliteDynamics.Epoch=SatelliteDynamics.Epoch("2018-01-01"),
    control_type::Symbol=:dipole, # one of :thrust :drag_ratio :torque or :dipole
    magnetic_model::Symbol=:IGRF13,
    add_solar_radiation_pressure::Bool=false,
    add_sun_thirdbody::Bool=false,
    add_moon_thirdbody::Bool=false,
    add_gravity_gradient_torque::Bool=false
)
    return OrbitDynamicsParameters(
        satellite_model,
        distance_scale,
        time_scale,
        control_scale,
        angular_rate_scale,
        n_gravity,
        m_gravity,
        start_epoch,
        control_type,
        magnetic_model,
        add_solar_radiation_pressure,
        add_sun_thirdbody,
        add_moon_thirdbody,
        add_gravity_gradient_torque
    )
end

function toDict(m::OrbitDynamicsParameters)
    return Dict(
        "satellite_model" => toDict(m.satellite_model),
        "distance_scale" => m.distance_scale,
        "time_scale" => m.time_scale,
        "control_scale" => m.control_scale,
        "angular_rate_scale" => m.angular_rate_scale,
        "n_gravity" => m.n_gravity,
        "m_gravity" => m.m_gravity,
        "start_epoch" => SatelliteDynamics.caldate(m.start_epoch),
        "control_type" => m.control_type,
        "magnetic_model" => m.magnetic_model,
        "add_solar_radiation_pressure" => m.add_solar_radiation_pressure,
        "add_sun_thirdbody" => m.add_sun_thirdbody,
        "add_moon_thirdbody" => m.add_moon_thirdbody,
        "add_gravity_gradient_torque" => m.add_gravity_gradient_torque,
    )
end

""" cartesian_acceleration_torque(x::Array{<:Real}, params::OrbitSimulationParameters, epc::Epoch)
Compute the accelerations and torque experienced by a satellite
"""
function cartesian_acceleration_torque(x::Array{<:Real}, u::Array{<:Real}, epc::SatelliteDynamics.Epoch, params::OrbitDynamicsParameters)

    r = x[1:3] # ECI frame position
    v = x[4:6] # Inertial frame velocity
    if length(x) > 6
        q = x[7:10] # Body to ECI quaternion
        ω = x[11:13] # Body frame angular rate
    else
        q = nothing
        ω = nothing
    end

    # Compute ECI to ECEF Transformation -> IAU2010 Theory
    PN = SatelliteDynamics.bias_precession_nutation(epc)
    E = SatelliteDynamics.earth_rotation(epc)
    W = SatelliteDynamics.polar_motion(epc)
    R = W * E * PN

    # eltype(x) makes this forward diff friendly
    a = zeros(eltype(x), 3) # inertial frame acceleration
    τ = zeros(eltype(x), 3) # body frame torque

    # spherical harmonic gravity
    a += SatelliteDynamics.accel_gravity(x, R, params.n_gravity, params.m_gravity)

    # thrust control
    if params.control_type == :thrust
        a += u
    end

    # torque control
    if params.control_type == :torque
        τ += u
    end

    # atmospheric drag
    if params.control_type == :drag_ratio || isnothing(q)
        if params.control_type == :drag_ratio
            drag_ratio = u[1]
        else
            drag_ratio = 0
        end
        a_drag_inertial = drag_ratio_acceleration(x, drag_ratio, epc, params)
    else
        a_drag_body, τ_drag = drag_acceleration_torque(x, epc, params) # body frame
        a_drag_inertial = Q(q) * a_drag_body
        τ += τ_drag
    end
    a += a_drag_inertial

    # magnetorquer torque
    if !isnothing(q)
        if params.control_type == :dipole
            m = u
        else
            m = zeros(3)
        end
        τ_mag = magnetic_torque(x, m, epc, params)
        τ += τ_mag
    end

    # gravity gradient
    if params.add_gravity_gradient_torque
        τ_gg = gravity_gradient_torque(x, params)
        τ += τ_gg
    end

    # Sun and Moon
    if !isnothing(q)
        if params.add_solar_radiation_pressure
            a_srp_body, τ_srp = solar_acceleration_torque(x, epc, params)
            a_srp_inertial = Q(q) * a_srp_body
            a += a_srp_inertial
            τ += τ_srp
        end
    end

    # third body sun
    if params.add_sun_thirdbody
        r_sun = SatelliteDynamics.sun_position(epc)
        a += SatelliteDynamics.accel_thirdbody_sun(x, r_sun)
    end

    # third body moon
    if params.add_moon_thirdbody
        r_moon = SatelliteDynamics.moon_position(epc)
        a += SatelliteDynamics.accel_thirdbody_moon(x, r_moon)
    end

    return a, τ
end

""" cartesian_keplarian_dynamics(x::Vector{<:Real})
The unperturbed keplarian cartesian orbit dynamics.
"""
function cartesian_keplarian_dynamics(x::Vector{<:Real}, u::Vector{<:Real}, ts::Real)
    r = x[1:3]
    rmag = sqrt(r'r)
    v = x[4:6]

    ṙ = v
    v̇ = -((SatelliteDynamics.GM_EARTH / rmag^3) * r)

    return [ṙ; v̇]
end


""" drag_acceleration_torque(airspeed, epc, params)
Compute the resultant forces and torques on a spacecraft due to drag.
See Markley and Crassidis pg 108

 * `airspeed` is the body-frame velocity of the spacecraft wrt the atmosphere (drag acts opposite to airspeed)
 * `epc` is the current epoch
 * `params::OrbitSimulationParameters` is a struct of simulation and model parameters

All units are SI kg-m-s, return values are in the body-frame
"""
function drag_acceleration_torque(x::Vector{<:Real}, epc::SatelliteDynamics.Epoch, params::OrbitDynamicsParameters)
    model = params.satellite_model

    q = x[7:10]

    airspeed_inertial = airspeed_from_state(x)
    airspeed = Q(q)' * airspeed_inertial

    force = zeros(eltype(airspeed), 3)
    torque = zeros(eltype(airspeed), 3)

    ρ = SatelliteDynamics.density_harris_priester(epc, x[1:3])
    CD = model.drag_coefficient
    CoM = model.center_of_mass

    for f in model.faces
        if airspeed'f.normal > 0.0
            f_force = -0.5 * ρ * CD * f.area * (airspeed'f.normal) * airspeed
            force += f_force
            torque += cross(f.position - CoM, f_force)
        end
    end
    accel = force / model.mass
    return accel, torque
end

""" drag_acceleration_torque(airspeed, epc, params)
Compute the resultant forces and torques on a spacecraft due to drag.
See Markley and Crassidis pg 108

 * `airspeed` is the body-frame velocity of the spacecraft wrt the atmosphere (drag acts opposite to airspeed)
 * `epc` is the current epoch
 * `params::OrbitSimulationParameters` is a struct of simulation and model parameters

All units are SI kg-m-s, return values are in the body-frame
"""
function drag_ratio_acceleration(x::Vector{<:Real}, drag_ratio::Real, epc::SatelliteDynamics.Epoch, params::OrbitDynamicsParameters)
    @assert 0 <= drag_ratio <= 1.0
    model = params.satellite_model

    v = airspeed_from_state(x)
    v_mag = norm(v)

    ρ = SatelliteDynamics.density_harris_priester(epc, x[1:3])
    C_D = model.drag_coefficient
    mass = model.mass
    A = model.min_drag_area + drag_ratio * (model.max_drag_area - model.min_drag_area)

    F_drag = -0.5 * ρ * C_D * A * v_mag .* v
    a_drag = F_drag / mass

    return a_drag
end

""" solar_acceleration_torque(x::Vector{<:Real}, epc::SatelliteDynamics.Epoch, params::OrbitSimulationParameters)
Compute the acceleration and torque due to solar radiation pressure on each face.
Assumes the SRP center of pressure is the geometric center of each face.
Does not account for self-shielding of faces, assumes constant distance between earth and sun (P_sun is constant).
See Montenbruk and Gill eq 3.73 and Markley and Crassidis eq 3.167
"""
function solar_acceleration_torque(x::Vector{<:Real}, epc::SatelliteDynamics.Epoch, params::OrbitDynamicsParameters)
    model = params.satellite_model

    r = x[1:3] # ECI frame position
    v = x[4:6] # ECI frame velocity
    q = x[7:10] # body to inertial quaternion

    r_sun = SatelliteDynamics.sun_position(epc)
    r_sun_body = Q(q)'r_sun
    s = r_sun_body / norm(r_sun_body)

    a = zeros(eltype(x), 3)
    τ = zeros(eltype(x), 3)

    nu = SatelliteDynamics.eclipse_conical([r; v], r_sun)
    P_sun = SatelliteDynamics.P_SUN

    for f in model.faces
        cθ = s'f.normal
        ϵ = f.reflectivity_coefficient_ϵ
        if cθ > 0.0
            force_srp = -nu * P_sun * f.area * cθ * ((1 - ϵ) * s + 2 * ϵ * cθ * f.normal)
            τ += cross(f.position - model.center_of_mass, force_srp)
            a += force_srp / model.mass
        end
    end

    return a, τ
end

""" airspeed_from_state(x)
Compute the inertial-frame airspeed wrt to the atmosphere given the satellite state `x`
"""
function airspeed_from_state(x)
    r = x[1:3] # ECI frame position
    v = x[4:6] # Inertial frame velocity

    ω_atmosphere = [0, 0, SatelliteDynamics.OMEGA_EARTH]
    airspeed_inertial = v + cross(ω_atmosphere, r)

    return airspeed_inertial
end

""" magnetic_torque()

"""
function magnetic_torque(x::Vector{<:Real}, m::Vector{<:Real}, epc::SatelliteDynamics.Epoch, params::OrbitDynamicsParameters)
    r = x[1:3]
    q = x[7:10]

    B_eci = magnetic_B_vector(r, epc, params)
    B_body = Q(q)'B_eci

    τ_mag = cross(m, B_body)
    return τ_mag
end

function magnetic_B_vector_body(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters)
    r = x[1:3]
    q = x[7:10]

    B_eci = magnetic_B_vector(r, t, params)
    B_body = Q(q)'B_eci

    return B_body
end

function magnetic_B_vector_body_dot(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters)
    r = x[1:3]
    v = x[4:6]
    B_eci = magnetic_B_vector(r, t, params)
    dB_eci_dr = ForwardDiff.jacobian(r_ -> magnetic_B_vector(r_, t, params), r)
    B_eci_dot = dB_eci_dr * v

    q = x[7:10]
    Q_body_eci = Q(q)' # transforms eci vectors to body

    B_body = Q_body_eci * B_eci

    ω_body = x[11:13] # Body frame angular rate

    B_dot_body = Q_body_eci * B_eci_dot + hat(B_body) * ω_body

    return B_dot_body
end

function magnetic_B_vector(r::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters)
    epc = sim_time_to_epoch(params, t)
    return magnetic_B_vector(r, epc, params)
end

function magnetic_B_vector(r::Vector{<:Real}, epc::SatelliteDynamics.Epoch, params::OrbitDynamicsParameters)
    if params.magnetic_model == :dipole
        return dipole_magnetic_field_B(r)
    elseif params.magnetic_model == :IGRF13
        return IGRF13(r, epc)
    else
        error("Magnetic model $(params.magnetic_model) unrecognized")
    end
end

function gravity_gradient_torque(x::Vector{<:Real}, params::OrbitDynamicsParameters)
    μ = SatelliteDynamics.GM_EARTH

    r_inertial = x[1:3]
    q_b2i = x[7:10]

    r_body = Q(q_b2i)'r_inertial

    r_mag = norm(r_body)
    n = -r_body / r_mag # body frame nadir pointing vector

    J = params.satellite_model.inertia

    τ_gg = (3 * μ / r_mag^3) * hat(n) * J * n

    return τ_gg
end

function satellite_orbit_attitude_dynamics(z::Array{<:Real}, p::OrbitDynamicsParameters, t::Real)
    zdot = zero(z)
    satellite_orbit_attitude_dynamics!(zdot, z, p, t)
    return zdot
end

function satellite_orbit_attitude_dynamics!(zdot::Array{<:Real}, z::Array{<:Real}, p, t::Real)
    dscale = p.distance_scale
    tscale = p.time_scale
    uscale = p.control_scale
    ωscale = p.angular_rate_scale

    # Unpack and unscale state vector
    r = z[1:3] * dscale # ECI frame position
    v = z[4:6] * (dscale / tscale) # Inertial frame velocity
    q = z[7:10] # Body to ECI quaternion
    ω = z[11:13] * (ωscale / tscale) # Body frame angular rate
    u = z[14:16] * uscale # Body frame magnetic dipole

    x = [r; v; q; ω]

    epc = sim_time_to_epoch(p, t)
    a, τ = cartesian_acceleration_torque(x, u, epc, p)

    J = p.satellite_model.inertia

    ṙ = v
    v̇ = a
    q̇ = 0.5 * L(q) * H * ω
    ω̇ = J \ (τ - cross(ω, J * ω))

    zdot[1:3] .= ṙ / (dscale / tscale)
    zdot[4:6] .= v̇ / (dscale / tscale^2)
    zdot[7:10] .= q̇ / (1 / tscale)
    zdot[11:13] .= ω̇ / (ωscale / tscale^2)
    zdot[14:16] .= zeros(eltype(u), 3)
end

function satellite_orbit_dynamics!(zdot::Array{<:Real}, z::Array{<:Real}, p, t::Real)
    dscale = p.distance_scale
    tscale = p.time_scale
    uscale = p.control_scale

    # Unpack and unscale state vector
    r = z[1:3] * dscale # ECI frame position
    v = z[4:6] * (dscale / tscale) # Inertial frame velocity
    u = z[7:9] * uscale

    # state vector for cartesian_acceleration_torque()
    x = [r; v]

    epc = sim_time_to_epoch(p, t)
    a, _ = cartesian_acceleration_torque(x, u, epc, p)

    ṙ = v
    v̇ = a

    zdot .= [ṙ / (dscale / tscale)
        v̇ / (dscale / tscale^2)
        zeros(eltype(u), 3)]
end

""" state_from_osc(xosc, q, ω)
convert osculating elements to ECI cartesian and concatenate with attitude and angular rate
"""
function state_from_osc(xosc::Vector{<:Real}, q::Vector{<:Real}, ω::Vector{<:Real})
    rv = SatelliteDynamics.sOSCtoCART(xosc, use_degrees=false)
    x = [rv[1:3]; rv[4:6]; q; ω]
    return x
end

function sim_time_to_epoch(model, t)
    return model.start_epoch + t
end

function null_controller(x::Vector{<:Real}, t, params::OrbitDynamicsParameters)
    return zeros(eltype(x), 3)
end

function simulate_satellite_orbit_attitude(x0::Array{<:Real}, params::OrbitDynamicsParameters, tspan::Tuple{<:Real,<:Real}; controller=null_controller, controller_dt=1.0)

    t0 = tspan[1]
    next_controller_update_time = t0 + controller_dt

    function controls_condition(z, t, integrator)
        params = integrator.p
        tscale = params.time_scale

        return ((t * tscale) - next_controller_update_time) / tscale
    end

    function controls_affect!(integrator)
        params = integrator.p

        dscale = params.distance_scale
        tscale = params.time_scale
        uscale = params.control_scale
        ωscale = params.angular_rate_scale

        z = integrator.u
        t = integrator.t * tscale

        # Unpack and unscale state vector
        r = z[1:3] * dscale # ECI frame position
        v = z[4:6] * (dscale / tscale) # Inertial frame velocity
        q = z[7:10] # Body to ECI quaternion
        ω = z[11:13] * (ωscale / tscale) # Body frame angular rate
        u = z[14:16] * uscale # Body frame magnetic dipole

        x = [r; v; q; ω]

        u = controller(x, t, params)
        integrator.u[14:16] .= u / uscale
        next_controller_update_time = t + controller_dt
    end

    controls_callback = ContinuousCallback(controls_condition, controls_affect!)

    u0 = controller(x0, t0, params)

    dscale = params.distance_scale
    tscale = params.time_scale
    uscale = params.control_scale
    ωscale = params.angular_rate_scale

    r0 = x0[1:3] # ECI frame position
    v0 = x0[4:6] # Inertial frame velocity
    q0 = x0[7:10] # Body to ECI quaternion
    ω0 = x0[11:13] # Body frame angular rate

    z0 = [
        r0 / dscale
        v0 / (dscale / tscale)
        q0
        ω0 / (ωscale / tscale)
        u0 / uscale]

    prob = ODEProblem(satellite_orbit_attitude_dynamics!, z0, tspan ./ tscale, params)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12, callback=controls_callback, progress=true)

    # unpack time, states, and controls
    thist = tscale .* sol.t
    xhist = [sol[1:3, :] * dscale
        sol[4:6, :] * (dscale / tscale)
        sol[7:10, :]
        sol[11:13, :] * (ωscale / tscale)
    ]
    uhist = sol[14:16, :] * uscale

    return xhist, uhist, thist
end

function simulate_satellite_orbit(x0::Array{<:Real}, params::OrbitDynamicsParameters, tspan::Tuple{<:Real,<:Real}; controller=null_controller, controller_dt=1.0)

    t0 = tspan[1]
    next_controller_update_time = t0 + controller_dt

    function controls_condition(z, t, integrator)
        params = integrator.p
        tscale = params.time_scale

        return ((t * tscale) - next_controller_update_time) / tscale
    end

    function controls_affect!(integrator)
        params = integrator.p

        dscale = params.distance_scale
        tscale = params.time_scale
        uscale = params.control_scale

        z = integrator.u
        t = integrator.t * tscale

        # Unpack and unscale state vector
        r = z[1:3] * dscale # ECI frame position
        v = z[4:6] * (dscale / tscale) # Inertial frame velocity
        u = z[7:9] * uscale # Body frame magnetic dipole

        x = [r; v]

        u = controller(x, t, params)
        integrator.u[7:9] .= u / uscale
        next_controller_update_time = t + controller_dt
    end

    controls_callback = ContinuousCallback(controls_condition, controls_affect!)

    u0 = controller(x0, t0, params)

    dscale = params.distance_scale
    tscale = params.time_scale
    uscale = params.control_scale

    r0 = x0[1:3] # ECI frame position
    v0 = x0[4:6] # Inertial frame velocity

    z0 = [
        r0 / dscale
        v0 / (dscale / tscale)
        u0 / uscale]

    prob = ODEProblem(satellite_orbit_dynamics!, z0, tspan ./ tscale, params)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12, callback=controls_callback, progress=true)

    # unpack time, states, and controls
    thist = tscale .* sol.t
    xhist = [sol[1:3, :] * dscale
        sol[4:6, :] * (dscale / tscale)
    ]
    uhist = sol[7:9, :] * uscale

    return xhist, uhist, thist
end


function rk4(dynamics, z::Array{<:Real}, params::OrbitDynamicsParameters, dt::Real, t::Real)
    # rk4 for integration
    k1 = dt * dynamics(z, params, t)
    k2 = dt * dynamics(z + k1 / 2, params, t + dt / 2)
    k3 = dt * dynamics(z + k2 / 2, params, t + dt / 2)
    k4 = dt * dynamics(z + k3, params, t + dt)
    return z + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end

function rk4(dynamics, x::Vector{<:Real}, u::Vector{<:Real}, dt::Real, t::Real)
    # rk4 for integration
    k1 = dt * dynamics(x, u, t)
    k2 = dt * dynamics(x + k1 / 2, u, t + dt / 2)
    k3 = dt * dynamics(x + k2 / 2, u, t + dt / 2)
    k4 = dt * dynamics(x + k3, u, t + dt)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end

function scaled_controller(controller, z::Array{<:Real}, params::OrbitDynamicsParameters, t::Real)
    dscale = params.distance_scale
    tscale = params.time_scale
    uscale = params.control_scale
    ωscale = params.angular_rate_scale

    t = t * tscale

    # Unpack and unscale state vector
    r = z[1:3] * dscale # ECI frame position
    v = z[4:6] * (dscale / tscale) # Inertial frame velocity
    q = z[7:10] # Body to ECI quaternion
    ω = z[11:13] * (ωscale / tscale) # Body frame angular rate

    x = [r; v; q; ω]

    u = controller(x, t, params) / uscale

    return u
end

function simulate_satellite_orbit_attitude_rk4(x0::Array{<:Real}, params::OrbitDynamicsParameters, tspan::Tuple{<:Real,<:Real}; integrator_dt=0.1, controller=null_controller, controller_dt=1.0)


    dscale = params.distance_scale
    tscale = params.time_scale
    uscale = params.control_scale
    ωscale = params.angular_rate_scale

    r0 = x0[1:3] # ECI frame position
    v0 = x0[4:6] # Inertial frame velocity
    q0 = x0[7:10] # Body to ECI quaternion
    ω0 = x0[11:13] # Body frame angular rate

    z0 = [
        r0 / dscale
        v0 / (dscale / tscale)
        q0
        ω0 / (ωscale / tscale)
    ]

    Nt = Int(ceil((tspan[2] - tspan[1]) / integrator_dt))

    thist = zeros(Nt)
    zhist = zeros((13, Nt))
    uhist = zeros((3, Nt))

    zhist[:, 1] = z0

    next_controller_update_time = tspan[1] / tscale

    @progress "Orbit Sim" for k = 1:Nt-1
        t = (integrator_dt * (k - 1) + tspan[1]) / tscale
        thist[k] = t

        if next_controller_update_time <= t
            uhist[:, k] = scaled_controller(controller, zhist[:, k], params, t)
            next_controller_update_time = t + (controller_dt / tscale)
        else
            uhist[:, k] = uhist[:, k-1]
        end


        znext = rk4(satellite_orbit_attitude_dynamics, [zhist[:, k]; uhist[:, k]], params, integrator_dt / tscale, t)
        znext[7:10] ./= norm(znext[7:10])
        zhist[:, k+1] .= znext[1:13]
    end
    # make all arrays the same length
    t = (integrator_dt * (Nt) + tspan[1]) / tscale
    thist[Nt] = t
    uhist[:, Nt] = uhist[:, Nt-1]

    # unpack time, states, and controls
    thist = thist .* tscale
    xhist = [zhist[1:3, :] * dscale
        zhist[4:6, :] * (dscale / tscale)
        zhist[7:10, :]
        zhist[11:13, :] * (ωscale / tscale)
    ]
    uhist = uhist .* uscale

    return xhist, uhist, thist
end

function monte_carlo_orbit_attitude(get_initial_state, controllers::Dict, Ntrials, params::OrbitDynamicsParameters, tspan::Tuple{<:Real,<:Real}; integrator_dt=0.1, controller_dt=1.0)
    Ntimesteps = Int(ceil((tspan[2] - tspan[1]) / integrator_dt))
    mc_data = Dict(
        key => Dict(
            "X" => zeros(Ntrials, 13, Ntimesteps),
            "U" => zeros(Ntrials, 3, Ntimesteps),
            "T" => zeros(Ntrials, 1, Ntimesteps))
        for (key, _) in controllers)

    if Threads.nthreads() < length(Sys.cpu_info())
        @warn "Use more threads! start Julia with --threads $(length(Sys.cpu_info())) to max out your CPU"
    end

    Threads.@threads for mc_step = 1:Ntrials
        x0 = get_initial_state()
        print("Thread $(Threads.threadid()), Trial $mc_step: x0 = $x0\n")
        for (controller_name, controller) in controllers
            xhist, uhist, thist = simulate_satellite_orbit_attitude_rk4(x0, params, tspan; integrator_dt=integrator_dt, controller=controller, controller_dt=controller_dt)
            mc_data[controller_name]["X"][mc_step, :, :] .= xhist
            mc_data[controller_name]["U"][mc_step, :, :] .= uhist
            mc_data[controller_name]["T"][mc_step, 1, :] .= thist
        end
    end

    return mc_data
end

function randbetween(min, max)
    return (max - min) * rand() + min
end

function mc_initial_orbital_elements(h_range, e_range, i_range, Ω_range, ω_range, M_range)
    return [
        randbetween(h_range[1], h_range[2]) + SatelliteDynamics.R_EARTH,
        randbetween(e_range[1], e_range[2]),
        randbetween(i_range[1], i_range[2]),
        randbetween(Ω_range[1], Ω_range[2]),
        randbetween(ω_range[1], ω_range[2]),
        randbetween(M_range[1], M_range[2])
    ]
end

function mc_initial_attitude()
    ϕ = rand(3)
    ϕ = ϕ / norm(ϕ)
    θ = randbetween(0, 2 * pi)
    r = θ * ϕ
    return axis_angle_to_quaternion(r)
end

function mc_initial_angular_velocity(ω_magnitude_range)
    ω_magnitude = randbetween(ω_magnitude_range[1], ω_magnitude_range[2])
    ω_direction = rand(3)
    ω_direction = ω_direction / norm(ω_direction)
    return ω_magnitude * ω_direction
end

function mc_setup_get_initial_state(h_range, e_range, i_range, Ω_range, ω_range, M_range, angular_rate_magnitude_range)
    function get_initial_state()
        x0_osc = mc_initial_orbital_elements(h_range, e_range, i_range, Ω_range, ω_range, M_range)
        q0 = mc_initial_attitude()
        ω0 = mc_initial_angular_velocity(angular_rate_magnitude_range)
        return state_from_osc(x0_osc, q0, ω0)
    end
    return get_initial_state
end

function propagate_orbit(x0, dt, Nt)

    orbit = [zeros(eltype(x0), length(x0)) for _ = 1:Nt+1]
    orbit[1] .= x0
    @progress "propagate_orbit" for k = 2:Nt+1
        orbit[k] .= rk4(cartesian_keplarian_dynamics, orbit[k-1], [0.0], dt, k * dt)
    end

    return orbit
end

function get_downsample(Ntrials, max_samples)
    sample_steps = Int(ceil(Ntrials / max_samples))
    downsample = [i % sample_steps == 0 for i = 0:Ntrials-1]
    downsample[end] = 1
    return downsample
end

function plot_position(thist, xhist; max_samples=1000, title="")
    downsample = get_downsample(length(thist), max_samples)
    plot(thist[downsample] / (60 * 60), xhist[1:3, downsample]' / 1e3, label=["x" "y" "z"])
    plot!(title=title, xlabel="Time (hours)", ylabel="Position (km)", linewidth=1.5)
end

function plot_position_3D(xhist; max_samples=1000, title="")
    downsample = get_downsample(size(xhist, 2), max_samples)
    xds = xhist[:, downsample]

    plot(xds[1, :], xds[2, :], xds[3, :])
end


function plot_velocity(thist, xhist; max_samples=1000, title="")
    downsample = get_downsample(length(thist), max_samples)
    plot(thist[downsample] / (60 * 60), xhist[4:6, downsample]' / 1e3, label=["v_x" "v_y" "v_z"])
    plot!(title=title, xlabel="Time (hours)", ylabel="Velocity (km/s)", linewidth=1.5)
end

function plot_attitude(thist, xhist; max_samples=1000, title="", q_desired=nothing)
    downsample = get_downsample(length(thist), max_samples)
    p = plot(thist[downsample] / (60 * 60), xhist[7:10, downsample]', label=["q1" "q2" "q3" "q4"], palette=palette(:tab10)[1:4])
    plot!(p, title=title, xlabel="Time (hours)", ylabel="Attitude", linewidth=1.5)
    if !isnothing(q_desired)
        if size(q_desired) == (4,)
            plot!(p, [thist[1], thist[end]] ./ (60 * 60), [q_desired q_desired]', label=["" "" "" ""], style=:dash, palette=palette(:tab10)[1:4])
        else
            Nt = size(q_desired, 1)
            qd_t = range(thist[1], thist[end], Nt)
            plot!(p, qd_t ./ (60 * 60), q_desired, label=["" "" "" ""], style=:dash, palette=palette(:tab10)[1:4])
        end
    end
    return p
end

function plot_attitude_error(thist, xhist, q_desired; max_samples=1000, title="")
    downsample = get_downsample(length(thist), max_samples)
    qds = xhist[7:10, downsample]
    q_error = hcat([L(q_desired)'qds[:, k] for k = axes(qds, 2)]...)'
    p = plot(thist[downsample] / (60 * 60), q_error, label=["q̃1" "q̃2" "q̃3" "q̃4"], palette=palette(:tab10)[1:4])
    plot!(p, title=title, xlabel="Time (hours)", ylabel="Error Quaternion", linewidth=1.5)
    return p
end

function plot_attitude_error_mrp(thist, xhist, q_desired; max_samples=1000, title="")
    downsample = get_downsample(length(thist), max_samples)
    qds = xhist[7:10, downsample]
    p_error = hcat([quaternion_to_mrp(L(q_desired)'qds[:, k]) for k = axes(qds, 2)]...)'
    p = plot(thist[downsample] / (60 * 60), p_error, label=["p̃1" "p̃2" "p̃3"], palette=palette(:tab10)[1:4])
    plot!(p, title=title, xlabel="Time (hours)", ylabel="Error MRP", linewidth=1.5)
    return p
end

function plot_rates(thist, xhist; max_samples=1000, title="")
    downsample = get_downsample(length(thist), max_samples)
    plot(thist[downsample] / (60 * 60), rad2deg.(xhist[11:13, downsample]'), label=["ω_x" "ω_y" "ω_z"])
    plot!(title=title, xlabel="Time (hours)", ylabel="Rates (deg/s)", linewidth=1.5)
end

function plot_controls(thist, uhist, params; max_samples=1000, title="")
    downsample = get_downsample(length(thist), max_samples)
    plot(thist[downsample] / (60 * 60), uhist[1:end, downsample]', label=["m_x" "m_y" "m_z"])
    plot!(title=title, xlabel="Time (hours)", ylabel="Controls (Am²)", linewidth=1.5, palette=palette(:tab10)[1:3])
    plot!(thist[downsample] / (60 * 60), params.satellite_model.max_dipoles' .* ones(length(thist[downsample])), label=["" "" ""], style=:dash, palette=palette(:tab10)[1:3])
    plot!(thist[downsample] / (60 * 60), -params.satellite_model.max_dipoles' .* ones(length(thist[downsample])), label=["" "" ""], style=:dash, palette=palette(:tab10)[1:3])
end

function list_to_matrix(list)
    return permutedims(hcat([list[k] for k = axes(list)[1]]...))
end