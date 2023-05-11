#################################### Low level for contour integrals

# Integration on complex plane
circ(z0,r,θ) = z0 + r*cis(2π*θ)
circle_path(z0,r,res) = circ.(z0,r,(0:1/res:1-1/res)) # array of coords of circle around z0
∇array(a,dx) = [a[mod1(i+1,length(a))]-a[mod1(i-1,length(a))] for i=1:length(a)]/(2*dx)

function test_circle_path()
    len = 30
    cp = circle_path(im+2,0.1,len)
    @assert len==length(cp)
    test_circle_path(cp)
end

function test_circle_path(cp)
    p = Plots.scatter(real(cp),imag(cp))
    display(p)
end

# A = z ↦ A(z) is an operator valued function, the integration contour is ∂B_r(z0), res is the number of points on the circle around, returns 1/(2πi) ∫ A(z) dz
# contour_integral(A,z0,r,res) = sum(A.(circle_path(z0,r,res)))*r/(res*im)

function contour_integral(A,z0,resolution,r=-1)
    res = Int(resolution)
    cp = circle_path(z0,abs(r),res)
    ∇cp = ∇array(cp,2π/res)
    l = [A(cp[i]) for i=1:res]
    sum(l.*∇cp)/(res*im)
end

# Evaluates (z-H)^(-1)
function resolvent(H,z,resolution,tests=false)
    Es,ψs = eigen(H)
    E = Es[1]; e = Es[2]
    P = ψs[:,1]*ψs[:,1]'
    @assert abs(1-norm(P)) + norm(P^2-P) + norm(P' - P) < 1e-10
    res(z) = inv(z*I-H)
    P_integrated = contour_integral(res,E,resolution,(e-E)/2)
    if tests
        px("norm P_int ",norm(P_integrated),", P_integrated = P ",dist(P_integrated,P)," norm E ",norm(res(E)))
    end
    P_integrated
end

function test_contour_integral()
    N = 20
    H = rand(N,N)
    H = H+H'
    Es,ψs = eigen(H)
    E = Es[1]
    resolvent(H,E,1000,true)
end
