import SatelliteDynamics
using Interpolations
import Convex, SCS #COSMO


include("satellite_simulator.jl")

function h_B_aligned_initial_conditions(x0::Vector{<:Real}, ω_magnitude_rad_s, params::OrbitDynamicsParameters)
    B0 = magnetic_B_vector(x0[1:3], 0.0, params) # inertial frame B
    b0 = B0 / norm(B0)
    h0 = eigen(params.satellite_model.inertia).vectors[:, end] # largest principle axis
    ω0 = params.satellite_model.inertia \ h0
    ω0 = ω_magnitude_rad_s * ω0 / norm(ω0)
    bperp = cross(b0, h0)
    bperp = bperp / norm(bperp)
    θ0 = acos(b0'h0)
    q0 = [sin(θ0 / 2); bperp * cos(θ0 / 2)]
    q0 = q0 / norm(q0)

    return [x0[1:6]; q0; ω0]
end

""" bcross_control(x, epc)
Detumble controller
See Markley and Crassidis eq 7.48, p 308
"""
function bcross_control(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters; B=magnetic_B_vector_body(x, t, params), k=1.0, saturate=true)
    Bnorm = norm(B)
    b = B / Bnorm
    ω = x[11:13]
    m = (k / Bnorm) * cross(ω, b)

    if saturate
        model = params.satellite_model
        m .= clamp.(m, -model.max_dipoles, model.max_dipoles)
    end

    return m
end

""" bcross_gain
Optimal gain for bcross_control, according to Avanzini and Giulietti 2012
"""
function bcross_gain(x_osc_0, params)
    Jmin = minimum(eigvals(params.satellite_model.inertia))
    Ω = 1 / sqrt(x_osc_0[1]^3 / SatelliteDynamics.GM_EARTH)
    ξ_m = x_osc_0[3] # this is inclination, should be mangetic inclination

    k_bcross = 2 * Ω * (1 + sin(ξ_m)) * Jmin # equation 30
end

Bmeasured = NaN * ones(3)
ticks_since = 0
function magnetic_control_sensed(controller, x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters; sensor_ticks=5)
    global ticks_since
    global Bmeasured

    if t == 0.0
        Bmeasured = NaN * ones(3)
        ticks_since = 0
    end

    if ticks_since % sensor_ticks == 0
        Bmeasured = magnetic_B_vector_body(x, t, params)
        ticks_since += 1
        return zeros(3)
    else
        ticks_since += 1
        return controller(x, t, params, Bmeasured)
    end
end

function boptimal_control(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters; B=magnetic_B_vector_body(x, t, params), k=1.0, saturate=true)
    Bnorm = norm(B)
    b = B / Bnorm
    ω = x[11:13]
    c = -hat(ω) * b # = (b'hat(ω))'
    max_dipoles = params.satellite_model.max_dipoles
    m = -max_dipoles .* tanh.(k * c)

    if saturate
        model = params.satellite_model
        m .= clamp.(m, -model.max_dipoles, model.max_dipoles)
    end

    return m
end

""" boptimal_control_ball()
Forces m to be within the ball m'm <= min(m_max)^2
"""
function boptimal_control_ball(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters; B=magnetic_B_vector_body(x, t, params), k=1.0, saturate=true)
    Bnorm = norm(B)
    b = B / Bnorm
    ω = x[11:13]
    c = -hat(ω) * b # = (b'hat(ω))'
    max_dipoles = params.satellite_model.max_dipoles
    mmax = minimum(max_dipoles)

    cnorm = norm(c)
    # m = -mmax .* (c ./ cnorm) .* tanh(k * cnorm)
    m = -max_dipoles .* (c ./ cnorm) .* tanh(k * cnorm)

    if saturate
        model = params.satellite_model
        m .= clamp.(m, -model.max_dipoles, model.max_dipoles)
    end

    return m
end

function bmomentum_control(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters; B=magnetic_B_vector_body(x, t, params), k=1.0, saturate=true)
    Bnorm = norm(B)
    b = B / Bnorm
    ω = x[11:13]
    h = params.satellite_model.inertia * ω
    c = -hat(h) * b
    max_dipoles = params.satellite_model.max_dipoles
    m = -max_dipoles .* tanh.(k * c)

    if saturate
        model = params.satellite_model
        m .= clamp.(m, -model.max_dipoles, model.max_dipoles)
    end

    return m
end

function projection_control(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters; B=magnetic_B_vector_body(x, t, params), ϵ=10e-4, k1=1.0, k2=1.0, saturate=true)
    ω = x[11:13]
    J = params.satellite_model.inertia
    Bnorm = norm(B)
    Jω = J * ω
    k = k1 * exp.(-k2 * norm((B') / Bnorm * (Jω / (norm(Jω) + ϵ))))
    m = (k / (Bnorm^2)) * hat(B)' * Jω
    if saturate
        model = params.satellite_model
        m .= clamp.(m, -model.max_dipoles, model.max_dipoles)
    end
    return m
end

function compute_B_vectors(params::OrbitDynamicsParameters, orbit, dt)
    Nt = length(orbit)
    B_vectors = [zeros(eltype(orbit[1]), 3) for _ = 1:Nt]

    @progress "compute_B_vectors" for k = 1:Nt
        epc = params.start_epoch + (k - 1) * dt
        B_vectors[k] .= magnetic_B_vector(orbit[k][1:3], epc, params)
    end

    return B_vectors
end

function interpolate_B_vectors(B_vectors, time_range)

    B_x_interp = scale(interpolate([B_k[1] for B_k in B_vectors], BSpline(Cubic(Line(OnGrid())))), time_range)
    B_y_interp = scale(interpolate([B_k[2] for B_k in B_vectors], BSpline(Cubic(Line(OnGrid())))), time_range)
    B_z_interp = scale(interpolate([B_k[3] for B_k in B_vectors], BSpline(Cubic(Line(OnGrid())))), time_range)

    function clampt(t)
        tc = clamp(t, time_range[1], time_range[end])
        if t != tc
            @warn "Clamped $t to $tc" maxlog = 1
        end
        return tc
    end

    function B_vec_interp(t)
        tc = clampt(t)
        return [B_x_interp(tc), B_y_interp(tc), B_z_interp(tc)]
    end

    return B_vec_interp
end

function blookahead_core(B1, B2, J, ω, k, α)
    B1norm = norm(B1)
    B2norm = norm(B1)

    b1 = B1 / B1norm
    b2 = B2 / B2norm

    b̄ = [hat(b1) hat(b2)]'

    Z = [I(3) zeros(3, 3); zeros(3, 6)]

    Q1 = Z * b̄ * b̄' * Z
    Q2 = α * b̄ * b̄'


    h = J * ω
    q1 = Z * b̄ * h
    q2 = α * b̄ * h

    m̄ = (I + Q1 + Q2) \ (q1 + q2)

    m = tanh.(k * m̄[1:3])
end

function setup_blookahead_control(x0::Vector{<:Real}, params::OrbitDynamicsParameters, dt_orbit, Nt_orbit)
    r0 = x0[1:3]
    v0 = x0[4:6]

    orbit_time = dt_orbit * (Nt_orbit)
    orbital_states = propagate_orbit([r0; v0], dt_orbit, Nt_orbit)
    B_vectors = compute_B_vectors(params, orbital_states, dt_orbit)
    B_interp = interpolate_B_vectors(B_vectors, 0:dt_orbit:orbit_time)


    function blookahead_control(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters; k=1.0, saturate=true, tlookahead=10 * 60, α=10)
        q = x[7:10]
        B1 = magnetic_B_vector_body(x, t, params)
        B1_inertial = B_interp(t)
        B2_inertial = B_interp(t + tlookahead)
        B1_test = Q(q)'B1_inertial
        B2 = Q(q)'B2_inertial

        ω = x[11:13]
        J = params.satellite_model.inertia

        m = blookahead_core(B1, B2, J, ω, k, α)

        if saturate
            model = params.satellite_model
            m .= clamp.(m, -model.max_dipoles, model.max_dipoles)
        end

        return m
    end

    return blookahead_control
end

function bderivative_control(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters; k=1.0, saturate=true, tderivative=10 * 60, α=10)
    v = x[4:6]
    q = x[7:10]
    B1 = magnetic_B_vector_body(x, t, params)
    dBdx = ForwardDiff.jacobian(x_ -> magnetic_B_vector_body(x_, t, params), x)
    dBdr = dBdx[1:3, 1:3]
    B1dot = dBdr * v

    B2 = B1 + tderivative * B1dot

    ω = x[11:13]
    J = params.satellite_model.inertia

    m = blookahead_core(B1, B2, J, ω, k, α)

    if saturate
        model = params.satellite_model
        m .= clamp.(m, -model.max_dipoles, model.max_dipoles)
    end

    return m
end

function normalized_magnetic_B_vector_body(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters)
    B = magnetic_B_vector_body(x, t, params)
    Bnorm = norm(B)
    return B / Bnorm
end

function bbarbalat_minVd(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters; k=1.0, saturate=true, tsolver=10, α=1e-4)

    v = x[4:6]
    q = x[7:10]
    b = normalized_magnetic_B_vector_body(x, t, params)
    dBdx = ForwardDiff.jacobian(x_ -> normalized_magnetic_B_vector_body(x_, t, params), x)
    dBdr = dBdx[1:3, 1:3]
    Bdot = dBdr * v
    model = params.satellite_model
    T = [I(3) tsolver * I(3)]

    umin = -model.max_dipoles
    umax = model.max_dipoles

    B̂ = hat(b)
    Ḃ̂ = hat(Bdot)

    n = 3
    u = Convex.Variable(3)
    u̇ = Convex.Variable(3)

    ϵ = 1

    ω = x[11:13]
    J = params.satellite_model.inertia
    h = J * ω
    h = h / norm(h)

    objective = -h'B̂ * u + α * Convex.sumsquares(u̇) + α * Convex.sumsquares(u)
    problem = Convex.minimize(objective)

    problem.constraints += (Convex.quadform(u, B̂'B̂) - h' * (B̂ * u̇ + Ḃ̂ * u) ≤ ϵ)
    problem.constraints += umin ≤ u
    problem.constraints += u ≤ umax
    problem.constraints += umin ≤ T * [u; u̇]
    problem.constraints += T * [u; u̇] ≤ umax

    Convex.solve!(problem, SCS.Optimizer; silent_solver=true)

    if Int(problem.status) != 1
        Convex.solve!(problem, SCS.Optimizer; silent_solver=false)
        @infiltrate
    end

    m = Convex.evaluate(u)
    ṁ = Convex.evaluate(u̇)

    m = m .* tanh(k * norm(h))

    if saturate
        model = params.satellite_model
        m .= clamp.(m, -model.max_dipoles, model.max_dipoles)
    end

    return m
end

"""
    From "A new variant of the B-dot control for spacecraft magnetic detumbling"
"""
function make_bdot_variant(time_step)

    buffer = [zeros(3), zeros(3), zeros(3), zeros(3)]

    function update_bdot_estimate(buffer, B, dt)
        # Use five point stencil to estimate Bdot

        Bdot = (3 * buffer[4] - 16 * buffer[3] + 36 * buffer[2] - 48 * buffer[1] + 25 * B) / (12 * dt)

        buffer[2:4] = buffer[1:3]
        buffer[1] = B

        return Bdot
    end

    function bdot_variant(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters; k=1.0, saturate=true)
        B = magnetic_B_vector_body(x, t, params)
        B̂ = normalized_magnetic_B_vector_body(x, t, params)
        B̂dot = update_bdot_estimate(buffer, B̂, time_step)

        ε = 1e-2
        Σ = Diagonal([ε, ε, ε]) + hat(B̂)

        M = -(k / norm(B)) * cross(B̂, inv(Σ) * B̂dot)

        if saturate
            model = params.satellite_model
            M .= clamp.(M, -model.max_dipoles, model.max_dipoles)
        end

        return M
    end

    return bdot_variant
end

function bdot_variant_autodiff(x::Vector{<:Real}, t::Real, params::OrbitDynamicsParameters; k=1.0, saturate=true)

    v = x[4:6]

    B = magnetic_B_vector_body(x, t, params)
    B̂ = normalized_magnetic_B_vector_body(x, t, params)
    dB̂dx = ForwardDiff.jacobian(x_ -> normalized_magnetic_B_vector_body(x_, t, params), x)
    dB̂dr = dB̂dx[1:3, 1:3]
    B̂dot = dB̂dr * v

    ε = 1e-2
    Σ = Diagonal([ε, ε, ε]) + hat(B̂)

    M = -(k / norm(B)) * cross(B̂, inv(Σ) * B̂dot)

    if saturate
        model = params.satellite_model
        M .= clamp.(M, -model.max_dipoles, model.max_dipoles)
    end

    return M
end

function plot_omega_cross_B(thist, xhist, params; max_samples=1000, title="")
    downsample = get_downsample(length(thist), max_samples)
    xds = xhist[:, downsample]
    tds = thist[downsample]
    normalize(v) = v ./ norm(v)
    omega_cross_b = hcat([cross(normalize(xds[11:13, i]), normalize(magnetic_B_vector_body(xds[:, i], tds[i], params))) for i = 1:size(tds)[1]]...)'

    plot(tds / (60 * 60), omega_cross_b)
    plot!(title=title, xlabel="Time (hours)", ylabel="ω × B (normalized)", label=["x" "y" "z"], linewidth=1.5)
end

function plot_B_body(thist, xhist, params; max_samples=1000, title="")
    downsample = get_downsample(length(thist), max_samples)
    xds = xhist[:, downsample]
    tds = thist[downsample]
    B = hcat([magnetic_B_vector_body(xds[:, i], tds[i], params) for i = 1:size(tds)[1]]...)'

    plot(tds / (60 * 60), B)
    plot!(title=title, xlabel="Time (hours)", ylabel="Body Frame B (T)", label=["x" "y" "z"], linewidth=1.5)
end

function plot_B_inertial(thist, xhist, params; max_samples=1000, title="")
    downsample = get_downsample(length(thist), max_samples)
    xds = xhist[:, downsample]
    tds = thist[downsample]
    B = hcat([magnetic_B_vector(xds[1:3, i], tds[i], params) for i = 1:size(tds)[1]]...)'

    plot(tds / (60 * 60), B)
    plot!(title=title, xlabel="Time (hours)", ylabel="Inertial Frame B (T)", label=["x" "y" "z"], linewidth=1.5)
end

function plot_angular_momentum(thist, xhist, params; max_samples=1000, title="")
    downsample = get_downsample(length(thist), max_samples)
    xds = xhist[:, downsample]
    tds = thist[downsample]
    h = hcat([(params.satellite_model.inertia * xds[11:13, i]) for i = 1:size(tds)[1]]...)'

    plot(tds / (60 * 60), h)
    plot!(title=title, xlabel="Time (hours)", ylabel="Angular momentum", label=["x" "y" "z"], linewidth=1.5)
end