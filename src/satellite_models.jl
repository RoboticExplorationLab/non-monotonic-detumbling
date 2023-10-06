using LinearAlgebra

struct SatelliteFace
    area::Real # (m^2)
    normal::Vector{<:Real}
    position::Vector{<:Real} # (m) relative to geometric center
    reflectivity_coefficient_ϵ::Real # between 0 and 1, 0 = totally absorbing, 1 = totally reflecting
end

function SatelliteFace(
    area,
    normal,
    position;
    reflectivity_coefficient_ϵ=0.1
)
    return SatelliteFace(
        area,
        normal,
        position,
        reflectivity_coefficient_ϵ
    )
end

function toDict(m::SatelliteFace)
    return Dict(
        "area" => m.area,
        "normal" => m.normal,
        "position" => m.position,
        "reflectivity_coefficient_ϵ" => m.reflectivity_coefficient_ϵ
    )
end

struct SatelliteModel
    # fixed parameters
    mass::Real # kg
    inertia::Matrix{<:Real}
    center_of_mass::Vector{<:Real} # (m)
    faces::Vector{SatelliteFace}
    drag_coefficient::Real # C_D, dimensionless
    min_drag_area::Real
    max_drag_area::Real
    max_dipoles::Vector{<:Real} # Am², the maximum dipole that can be produced along each axis
    magnetometer_std_dev::Real
    gyro_std_dev::Real
    gyro_bias::Vector{<:Real}
    gyro_bias_limit::Real
end

Base.copy(m::SatelliteModel) = SatelliteModel(
    m.mass,
    m.inertia,
    m.center_of_mass,
    m.faces,
    m.drag_coefficient,
    m.min_drag_area,
    m.max_drag_area,
    m.max_dipoles,
    m.magnetometer_std_dev,
    m.gyro_std_dev,
    m.gyro_bias,
    m.gyro_bias_limit,
)
function toDict(m::SatelliteModel)
    return Dict(
        "mass" => m.mass,
        "inertia" => m.inertia,
        "center_of_mass" => m.center_of_mass,
        "faces" => [toDict(face) for face in m.faces],
        "drag_coefficient" => m.drag_coefficient,
        "min_drag_area" => m.min_drag_area,
        "max_drag_area" => m.max_drag_area,
        "max_dipoles" => m.max_dipoles,
        "magnetometer_std_dev" => m.magnetometer_std_dev,
        "gyro_std_dev" => m.gyro_std_dev,
        "gyro_bias" => m.gyro_bias,
        "gyro_bias_limit" => m.gyro_bias_limit,
    )
end

sat_dipole_magnitude(N, I, A) = N * I * A # N turns, I amps, A area

function coil_resistance(N, length, width, trace_area)
    copper_resistivity = 1.77e-8 # Ohm-meters
    perimeter = 2 * (length + width)
    return copper_resistivity * N * perimeter / trace_area
end

function magnetic_torque_magnitude_range(sat_dipole_magnitude, altitude)
    r = altitude + R_EARTH
    r_equatorial = [r, 0.0, 0.0]
    r_polar = [0.0, 0.0, r]

    τ_min = magnetic_torque_magnitude(sat_dipole_magnitude, r_equatorial)
    τ_max = magnetic_torque_magnitude(sat_dipole_magnitude, r_polar)

    return τ_min, τ_max
end

# PyCubed-Mini pocketqube
pqmini_faces = [
    SatelliteFace(0.06 * 0.066, [0.0, 1.0, 0.0], [0.0, 0.05 / 2, 0.0]) # Top +Y - projected area from -Y board
    SatelliteFace(0.06 * 0.066, [0.0, -1.0, 0.0], [0.0, -0.05 / 2, 0.0]) # Bottom -Y
    SatelliteFace(0.05^2, [0.0, 0.0, 1.0], [0.0, 0.0, 0.05 / 2]) # Side +Z
    SatelliteFace(0.05^2, [0.0, 0.0, -1.0], [0.0, 0.0, -0.05 / 2]) # Side -Z
    SatelliteFace(0.05^2, [1.0, 0.0, 0.0], [0.05 / 2, 0.0, 0.0]) # Side +X
    SatelliteFace(0.05^2, [-1.0, 0.0, 0.0], [-0.05 / 2, 0.0, 0.0]) # Side -X
]

pqmini_dipole_magnitude = let
    pqmini_voltage = 3.7
    pqmini_trace_width = 0.19e-3 # 7.5mil
    pqmini_trace_thickness = 0.036e-3 # 1oz copper
    pqmini_trace_area = pqmini_trace_thickness * pqmini_trace_width

    pqmini_N_turns = 154
    pqmini_coil_R = coil_resistance(pqmini_N_turns, 0.1, 0.1, pqmini_trace_area)
    pqmini_coil_I = pqmini_voltage / pqmini_coil_R

    sat_dipole_magnitude(pqmini_N_turns, pqmini_coil_I, 0.05 * 0.05)
end

pqmini_inertia_matrix = let
    inertia_CAD = [
        3.28E+05 -3853.031 1060.576
        -3853.031 3.18E+05 22.204
        1060.576 22.204 3.33E+05] # g*mm^2, determined from Fusion360 CAD model

    mass_CAD = 642.629 # g
    mass_measured = 191 # g
    g2kg = 1e3
    mm2m = 1e3
    inertia_matrix_kgm2 = (mass_measured .* inertia_CAD ./ mass_CAD) / g2kg / mm2m
end

pqmini_model = SatelliteModel(
    0.191,
    pqmini_inertia_matrix, # kg*m^2 determined from CAD
    [0.0, 0.0, 0.0],
    pqmini_faces,
    2.2, # CD
    0.05^2, # min drag area
    0.06 * 0.066, # max drag area
    pqmini_dipole_magnitude * ones(3),
    15e-9, # nT - source: GomSpace M315
    deg2rad(0.005) * sqrt(10), # 0.005 - source: MPU 3300
    zeros(3),
    deg2rad(1.0),
)

py4_faces = [
    SatelliteFace(0.1^2, [0.0, 0.0, 1.0], [0.0, 0.0, 0.15 / 2]) # Top +Z
    SatelliteFace(0.1^2, [0.0, 0.0, -1.0], [0.0, 0.0, -0.15 / 2]) # Bottom -Z
    SatelliteFace(0.1 * 0.15, [0.0, 1.0, 0.0], [0.0, 0.05, 0.0]) # Side +Y
    SatelliteFace(0.1 * 0.15, [0.0, -1.0, 0.0], [0.0, -0.05, 0.0]) # Side -Y
    SatelliteFace(0.1 * 0.15, [1.0, 0.0, 0.0], [0.05, 0.0, 0.0]) # Side +X
    SatelliteFace(0.1 * 0.15, [-1.0, 0.0, 0.0], [-0.05, 0.0, 0.0]) # Side -X
    SatelliteFace(0.2 * 0.15, [0.0, 1.0, 0.0], [0.15, 0.05, 0.0]) # Solar 1 +Y face
    SatelliteFace(0.2 * 0.15, [0.0, -1.0, 0.0], [0.15, 0.05, 0.0]) # Solar 1 -Y face
    SatelliteFace(0.2 * 0.15, [0.0, 1.0, 0.0], [-0.15, -0.05, 0.0]) # Solar 2 +Y face
    SatelliteFace(0.2 * 0.15, [0.0, -1.0, 0.0], [-0.15, -0.05, 0.0]) # Solar 2 -Y face
]

py4_dipole_magnitude = let
    py4_voltage = 5.06

    py4_N_turns = [40, 28 * 2, 28]
    py4_coil_R = [6.1 * 2, 8.4 * 2, 7.2 * 2] # Ohms
    py4_coil_I = py4_voltage ./ py4_coil_R # Amps
    py4_coil_A = [4218e-6, 3150e-6, 7091e-6] # m^2

    [sat_dipole_magnitude(py4_N_turns[i], py4_coil_I[i], py4_coil_A[i]) for i = 1:3]
end

py4_mass_measured = 1.58 # kg
py4_inertia_matrix = let
    inertia_CAD = [0.0043 -0.0003 0; -0.0003 0.0049 0; 0 0 0.0035] # kg*m^2, determined from CAD
    mass_CAD = 1.504 # kg
    py4_mass_measured .* inertia_CAD ./ mass_CAD
end

py4_model = SatelliteModel(
    py4_mass_measured, # mass
    py4_inertia_matrix, # inertia
    [0.0, 0.0, 0.0], # center of mass offset from geometric center
    py4_faces, # faces for comuting drag
    2.2, # coefficient of drag
    0.1^2, # min drag area
    (2 * 0.2 * 0.15 + 0.1 * 0.15), # max drag area
    py4_dipole_magnitude,
    15e-9, # T - source: GomSpace M315
    deg2rad(0.005) * sqrt(10), # 0.005 - source: MPU 3300
    zeros(3),
    deg2rad(1.0),
)

py4_model_no_noise = SatelliteModel(
    py4_mass_measured, # mass
    py4_inertia_matrix, # inertia
    [0.0, 0.0, 0.0], # center of mass offset from geometric center
    py4_faces, # faces for comuting drag
    2.2, # coefficient of drag
    0.1^2, # min drag area
    (2 * 0.2 * 0.15 + 0.1 * 0.15), # max drag area
    py4_dipole_magnitude,
    0.0,
    0.0,
    zeros(3),
    0.0,
)

py4_model_nodrag = SatelliteModel(
    py4_model.mass,
    py4_model.inertia,
    py4_model.center_of_mass,
    py4_model.faces,
    0.0, # zero drag
    py4_model.min_drag_area,
    py4_model.max_drag_area,
    py4_model.max_dipoles,
    15e-9, # nT - source: GomSpace M315
    deg2rad(0.005) * sqrt(10), # 0.005 - source: MPU 3300
    zeros(3),
    deg2rad(1.0),
)

dove_faces = [
    SatelliteFace(0.1^2, [0.0, 0.0, 1.0], [0.0, 0.0, 0.3 / 2]) # Top +Z
    SatelliteFace(0.1^2, [0.0, 0.0, -1.0], [0.0, 0.0, -0.3 / 2]) # Bottom -Z
    SatelliteFace(0.1 * 0.3, [0.0, 1.0, 0.0], [0.0, 0.05, 0.0]) # Side +Y
    SatelliteFace(0.1 * 0.3, [0.0, -1.0, 0.0], [0.0, -0.05, 0.0]) # Side -Y
    SatelliteFace(0.1 * 0.3, [1.0, 0.0, 0.0], [0.05, 0.0, 0.0]) # Side +X
    SatelliteFace(0.1 * 0.3, [-1.0, 0.0, 0.0], [-0.05, 0.0, 0.0]) # Side -X
    SatelliteFace(0.3 * 0.3, [1.0, 0.0, 0.0], [0.05, (0.05 + 0.3), 0.0]) # Solar 1 +X face
    SatelliteFace(0.3 * 0.3, [-1.0, 0.0, 0.0], [0.05, (0.05 + 0.3), 0.0]) # Solar 1 -X face
    SatelliteFace(0.3 * 0.3, [1.0, 0.0, 0.0], [0.05, -(0.05 + 0.3), 0.0]) # Solar 2 +X face
    SatelliteFace(0.3 * 0.3, [-1.0, 0.0, 0.0], [0.05, -(0.05 + 0.3), 0.0]) # Solar 2 -X face
]

dove_dipole_magnitude = let
    # these are not based on any knowledge of the dove satellites

    dove_voltage = 3.7 * 4
    dove_trace_width = 0.19e-3 # 7.5mil
    dove_trace_thickness = 0.036e-3 # 1oz copper
    dove_trace_area = dove_trace_thickness * dove_trace_width

    # assume coil resistances made to match dipoles
    dove_N_turns = 200
    dove_coil_R = coil_resistance(dove_N_turns, 0.1, 0.1, dove_trace_area)
    dove_coil_I = dove_voltage / dove_coil_R

    sat_dipole_magnitude(dove_N_turns, dove_coil_I, 0.1 * 0.1)
end

dove_model = SatelliteModel(
    6.0,
    diagm([1.0, 2.0, 3.0]),
    [0.0, 0.0, 0.0],
    dove_faces,
    2.2,
    0.1^2,
    (2 * 0.3 * 0.3 + 0.1 * 0.3),
    dove_dipole_magnitude * ones(3),
    15e-9, # nT - source: GomSpace M315
    deg2rad(0.005) * sqrt(10), # 0.005 - source: MPU 3300
    zeros(3),
    deg2rad(1.0),
)

# parameters based on Magnetorquer-Only Attitude Control paper
no_drag_1U = SatelliteModel(
    0.75,
    0.75 * (0.1^3 / 6) * diagm([1.0, 1.0, 1.0]),
    [0.0, 0.0, 0.0],
    [],
    0.0,
    0.0,
    0.0,
    0.19 * ones(3),
    0.0,
    0.0,
    zeros(3),
    deg2rad(0.0),
)

# parameters based on Magnetorquer-Only Attitude Control paper
no_drag_3U = SatelliteModel(
    3.5,
    diagm([0.04939, 0.04939, 0.005256]), # reordered axes so Z axis points out 10cm X 10cm face
    [0.0, 0.0, 0.0],
    [],
    0.0,
    0.0,
    0.0,
    [0.57, 0.57, 0.19],
    0.0,
    0.0,
    zeros(3),
    deg2rad(0.0),
)

kevin_sat = SatelliteModel(
    1.0,
    [1.959e-4 2016.333e-9 269.176e-9
        2016.333e-9 1.999e-4 2318.659e-9
        269.176e-9 2318.659e-9 1.064e-4],
    [0.0, 0.0, 0.0],
    [],
    0.0,
    0.0,
    0.0,
    [8.8e-3, 1.373e-2, 8.2e-3],
    0.0,
    0.0,
    zeros(3),
    deg2rad(0.0),
)

py4_model_diagonal = copy(py4_model)
py4_model_diagonal.inertia .= Matrix(diagm([0.001, 0.003, 0.005]))
py4_model_no_noise_diagonal = copy(py4_model_no_noise)
py4_model_no_noise_diagonal.inertia .= Matrix(diagm([0.001, 0.003, 0.005]))
