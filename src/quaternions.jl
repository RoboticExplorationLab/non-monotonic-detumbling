using LinearAlgebra
""" hat(ω)
The hat map, mapping vectors in R³ to the lie algebra so(3), the space of skew-symmetric matrices.
"""
function hat(ω)
    return [0 -ω[3] ω[2]
        ω[3] 0 -ω[1]
        -ω[2] ω[1] 0]
end

""" L(q)
Left quaternion product: L(q1)q2 = q1*q2 where * is the quaternion product.
"""
function L(q)
    [q[1] -q[2:4]'; q[2:4] q[1]*I+hat(q[2:4])]
end

""" R(q)
Right quaternion product: R(q2)q1 = q1*q2 where * is the quaternion product.
"""
function R(q)
    [q[1] -q[2:4]'; q[2:4] q[1]*I-hat(q[2:4])]
end

""" H
Maps 3 vectors to pure quaternions: H: v ∈ R³ -> [0;v] ∈ R⁴
"""
const H = [zeros(1, 3); I];

""" Q(q)
Rotation matrix from quaternion
"""
function Q(q)
    return H' * L(q) * R(q)' * H
end

""" quaternion_error(q, qref)
"""
function quaternion_error(q, qref)
    δq = L(qref)'q
    ϕ = (1 / δq[1]) * δq[2:4]
    return ϕ
end

function quaternion_error_jacobian(qref)
    return L(qref) * H
end

""" quaternion_geodesic_distance(q, qref)
equivalent to min(q'qref + 1, -q'qref + 1)
returns a number between 0 and 2 corresponding to the geodesic distance
between q and qref.
"""
function quaternion_geodesic_distance(q, qref)
    return -sign(q'qref) * (q'qref) + 1
end

""" gradient_quaternion_geodesic_distance(q, qref)
gradient with respect to q
"""
function gradient_quaternion_geodesic_distance(q, qref)
    return -sign(q'qref) * (quaternion_error_jacobian(q)'qref)
end

""" hessian_quaternion_geodesic_distance(q, qref)
hessian with respect to q
"""
function hessian_quaternion_geodesic_distance(q, qref)
    return sign(q'qref) * (qref'q) .* I(3)
end

""" mrp_to_quaternion(p)
Compute the quaternion for a MRP vector p
"""
function mrp_to_quaternion(p)
    np2 = p'p # ||p||^2
    s = 1 - np2
    v = 2p
    q = 1 / (1 + np2) * [s; v]
    return q
end

""" quaternion_to_mrp(q)
Compute the Modified Rodriguez Parameter of `q`
"""
function quaternion_to_mrp(q)
    s = q[1]
    v = q[2:4]

    return v ./ (1 + s)
end

function axis_angle_to_quaternion(r)
    θ = norm(r)
    return [cos(θ / 2); r * sinc(θ / (2 * pi)) * pi / 2] # sinc(θ/(2π)) = sin(πθ/2π) / πθ/2 = 2 sin(θ/2)/ πθ
end

""" normalize(v)
Normalize the vector `v` to be of unit length.
"""
function normalize(v)
    return v ./ sqrt(v'v)
end

""" vectors_to_attitude(B1, N1)
Given a body-frame vector `B1` and inertial frame vector `N1` that we wish to align
return the body-to-inertial quaternion representing that attitude.

Ref: https://www.xarg.org/proof/quaternion-from-two-vectors/
"""
function vectors_to_attitude(B, N; tol=1e-4)
    B = normalize(B)
    N = normalize(N)

    d = B'N
    w = cross(B, N)

    s = 1 + d
    if s < tol
        # B = -N, vectors are 180 degrees apart
        # many possible quaternions, but this is one 
        w = [-B[3], B[2], B[1]]
    end
    q = normalize([s; w])
    return q
end

""" vectors_to_attitude(B1, N1, B2, N2)
Given two body-frame and two inertial-frame vectors that we wish to align,
return the body to inertial quaternion representing that attitude.
The vectors should not be parallel.
`B1` and `N1`` will be aligned exactly.
The components of `B2` and `N2` that are orthogonal to `N1` will be aligned exactly

"""
function vectors_to_attitude(B1, N1, B2, N2)
    B1 = normalize(B1)
    N1 = normalize(N1)
    B2 = normalize(B2)
    N2 = normalize(N2)

    # rotate to align B1 and N1
    q1 = vectors_to_attitude(B1, N1)

    # rotate B2 into new frame
    B2 = Q(q1) * B2

    # then rotate around N1
    Project_N1 = I - N1 * N1' # projection onto space orthogonal to N1
    B2_ = Project_N1 * B2
    N2_ = Project_N1 * N2

    q2 = vectors_to_attitude(B2_, N2_)

    q_final = L(q1) * q2

    return q_final
end