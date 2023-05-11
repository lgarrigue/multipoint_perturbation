# This code contains the structures and functions enabling to create and apply multipoint perturbation up to second order and standard perturbation up to third order
using LinearAlgebra
using MKL
using Optim
include("contour_integration.jl")

px = println

# Main structure storing everything we need

mutable struct Objects
    Nh # size of Hilbert's space
    H0 # the base operator, it is -Δ for Schrödinger operators
    Gs # the points on which we assume to know the true solutions
    G # the point on which we want to know the density matrix
    n # number of Gj's
    Gij # differences Gi - Gj
    Es # ground state energy for each H0 + Gj
    allEs # all energies, allEs[1] is the list of all energies of H0+G1
    Ps # projections on ground states, for each H0 + Gj
    Kij # (Ei - (H0 + Gj))^(-1)_⟂ (we don't take the ground state)
    Kij_sing # (Ei - (H0 + Gj))^(-1) (we take all states)
    Rs # resolvents (z-(H0+Gj))^(-1)
    F_exact # density matrix solution at G
    E_exact # exact energy at G
    Fn # firsts orders of multipoint the expansion, Fn[(0,0,1)] for instance

    # Quantities with α
    αs # the list of α's
    α_dot_G # sum αj Gj
    Sα # sum_j αj
    δα # 1 - Sα
    δG # max(dist(Gi,Gj))
    δGmin # min(dist(Gi,Gj))
    g # G - sum_j αj Gj
    δg # norm(g)
    min_method # minimization method
    with_PT # if true, forces that δG = 0 and g=0 and computes with PT the second step
    G_with_PT

    # Misc
    path_plots
    rel_dist # true if we take relative distances
    operator_for_energy_norm # operator (H0 + 1)^1/2 for taking the energy norm
    operator_for_dual_energy_norm # operator (H0 + 1)^-1/2 for taking the dual of the energy norm
    function Objects(H0,Gs,rel;αs=-1,G=-1,coef_δα_minimization="infinity",limits_α_search=3,with_PT=false,G_with_PT=0) # either gives G, either αs. If one gives G (directly the G one which we want an approximation), we will optimize to get αs such that G is close to sum_j αj Gj, otherwise we take G = sum_j αj Gj
        O = new()
        O.rel_dist = rel
        O.with_PT = with_PT
        O.G_with_PT = G_with_PT

        # Loads H0, Gs
        O.Nh = size(H0,1)
        O.H0 = H0
        O.Gs = Gs
        O.n = length(Gs)

        # Prepare quantities for norms
        O.operator_for_energy_norm = sqrt(O.H0 + I)
        O.operator_for_dual_energy_norm = inv(O.operator_for_energy_norm)

        # Loads αs
        @assert (αs==-1 && G!=-1) || (αs!=-1 && G==-1)
        if αs==-1
            O.αs = minimization_αG(G,O.Gs;coef=coef_δα_minimization,limits_α_search=limits_α_search)
        else
            O.αs = αs
        end
        
        # Checks that if we put G which is on the affine space, then the minimization finds the same
        if αs==-1 && abs(sum(αs)-1) < 1e-10
            @assert dist(αs,O.αs;rel=true) < 1e-10
        end

        O.α_dot_G = sum(O.αs .* O.Gs)
        O.G = (G==-1) ? O.α_dot_G : G

        if O.with_PT
            O.αs = minimization_αG(O.G,O.Gs;coef="infinity")
            O.α_dot_G = sum(O.αs .* O.Gs)
            O.G = O.α_dot_G
            # Forces G to be on Aff Gj
            @assert abs(sum(O.αs) - 1) < 1e-10
        end

        O.Sα = sum(O.αs)
        O.δα = 1-O.Sα
        O.g = (G==-1) ? zero(O) : O.G - O.α_dot_G
        O.δg = norm(O.g)

        # Gij
        O.Gij = [copy(zeros(ComplexF64,O.Nh,O.Nh)) for i=1:O.n, j=1:O.n]
        for i=1:O.n, j=1:O.n
            if i != j
                O.Gij[i,j] = O.Gs[i] - O.Gs[j]
            end
        end
        ds = [sqrt(abs(O.αs[i]*O.αs[j]))*dist(O.Gs[i],O.Gs[j],O,-1) for i=1:O.n, j=1:O.n]
        O.δG = maximum(ds)
        O.δGmin = minimum(ds)

        # Diagonalize all H0 + Gj
        O.allEs = []
        ψss = []
        for j=1:O.n
            Es0,ψs0 = first_eigenmodes_of_matrix(H0 + Gs[j])
            push!(O.allEs,Es0)
            push!(ψss,ψs0)
        end

        # Es, Ps of ground level
        O.Ps = []; O.Es = []
        for j=1:O.n
            push!(O.Es,O.allEs[j][1])
            ψ = ψss[j][:,1]
            push!(O.Ps,ψ*ψ')
        end

        # Ks
        O.Kij = [copy(zero(O)) for i=1:O.n, j=1:O.n]
        O.Kij_sing = [copy(zero(O)) for i=1:O.n, j=1:O.n]
        for j=1:O.n
            ψs_j = ψss[j]
            allPs = [ψs_j[:,k]*ψs_j[:,k]' for k=1:O.Nh] # all projections of H0 + Gj, to compute K
            for i=1:O.n, k=1:O.Nh
                if i != j
                    O.Kij_sing[i,j] .+= (1/(O.Es[i] - O.allEs[j][k]))*allPs[k]
                end
                if k >=2 
                    O.Kij[i,j] .+= (1/(O.Es[i] - O.allEs[j][k]))*allPs[k]
                end
            end
        end

        # Resolvent functions
        O.Rs = []
        for j=1:O.n
            push!(O.Rs,z->inv(I*z-(O.H0 + O.Gs[j])))
        end

        # Prints info
        # px("Ground energies ",O.Es)
        E1s = [O.allEs[j][2] for j=1:O.n]
        # px("First excited energies ",E1s) 
        # we need to have max(E0s) < min(E1s) to satisfy the gap condition
        if maximum(O.Es) < minimum(E1s)
            # px("The necessary gap is ok")
        else
            px("The necessary gap is not ok, the computation is NOT valid")
        end

        # Forms G := ∑_(j=1)^n αj Gj and solves it
        O.F_exact, O.E_exact = ground_projection(O.H0+ (O.with_PT ? O.G_with_PT : O.G))

        # Zeroth order of G
        O.Fn = Dict()
        O.Fn[(0,0,0)] = (1/O.Sα)*sum(O.αs .* O.Ps)

        # Misc
        O.path_plots = "figs/"
        create_dir(O.path_plots)
        px("Computation of δα ",O.δα," δg ",O.δg," δG ",O.δG)

        O
    end
end

zero(O,complex=true) = complex ? zeros(ComplexF64,O.Nh,O.Nh) : zeros(O.Nh,O.Nh)

create_dir(path) = if !isdir(path) mkdir(path) end

function first_eigenmodes_of_matrix(H) # eigenmodes
    E,ψ = eigen(Hermitian(H))
    @assert abs(norm(ψ[:,1])-1) < 1e-5
    E, ψ # [:,1:p]
end

function ground_projection(H) # solves for one Hamiltonian H
    Es0,ψs0 = first_eigenmodes_of_matrix(H)
    ψ = ψs0[:,1]
    ψ*ψ', Es0[1]
end

function dist(f,g;rel=true,A=-1) # relative (or not) distance between f and g, whatever they are
    B = 1
    if A!=-1
        B = A
    end
    nor = (norm(B*f*B) + norm(B*g*B))/2
    if abs(nor) < 1e-15
        px("Division by zero in distance")
        return 0
    end
    norm(B*(f .- g)*B)/(rel ? nor : 1)
end

function dist(f,g,Ob,energy_norm=0)
    A = 1
    if energy_norm == 1
        A = Ob.operator_for_energy_norm
    elseif energy_norm == -1
        A = Ob.operator_for_dual_energy_norm
    end
    dist(f,g;rel=Ob.rel_dist,A=A)
end

function rand_herm(N) # random hermitian matrix
    M = rand(N,N)
    M + M'
end

# gives the minimizer α of D(G,α⋅Gs)^2 + coef |1-sum(α)|
function minimization_αG(G,Gs;coef="infinity",limits_α_search=2)
    # coeff = coef=="infinity" ? 1000 : coef
    coeff = coef
    n = length(Gs)
    nn = n + (coeff=="infinity" ? -1 : 0)
    function sumG(α)
        if coeff!="infinity"
            return sum(α .*Gs)
        end
        sum(α[j] .*Gs[j] for j=1:nn) + (1-sum(α))*Gs[n]
    end
    f(α) = dist(G, sumG(α);rel=true)^2 + (coeff=="infinity" ? 0 : coeff*abs2(sum(α)-1))
    α0 = zeros(nn)
    lower = -limits_α_search*ones(nn)
    upper = limits_α_search*ones(nn)
    sol = optimize(f, lower,upper,α0)
    # sol = optimize(f,α0)
    α_inter = Optim.minimizer(sol)
    α_min = coeff=="infinity" ? vcat(α_inter,[1-sum(α_inter)]) : α_inter
    α_min
end

############## Compute perturbation elements for multipoint perturbation

# Compute the elements with 2 R's
function elements_2_Rs(a,b,A,Ob) # computes (2πi)^(-1) ∮ Ra A Rb
    # K = a==b ? Ob.Kij : Ob.Kij_sing
    K = Ob.Kij
    Ob.Ps[a]*A*K[a,b] + K[b,a]*A*Ob.Ps[b]
end

perms3(Γ1,A,B,Γ2) = Γ1*A*Γ1*B*Γ2 + Γ1*A*Γ2*B*Γ1 + Γ2*A*Γ1*B*Γ1

# Compute the elements with 3 R's
function elements_3_Rs2(a,b,c,A,B,Ob) # computes (2πi)^(-1) ∮ Ra A Rb B Rc 
    Pa = Ob.Ps[a]
    Pb = Ob.Ps[b]
    Pc = Ob.Ps[c]
    K = Ob.Kij_sing
    Kr = Ob.Kij
    if a==b && b==c
        return -1*perms3(Pa,A,B,(Kr[a,a])^2) + perms3(Kr[a,a],A,B,Pa)
    elseif a != b && b != c && a !=c
        return Pa*A*K[a,b]*B*K[a,c] + K[b,a]*A*Pb*B*K[b,c] + K[c,a]*A*K[c,b]*B*Pc
    elseif a==b && b !=c
        return -Pa*A*Pa*B*(K[a,c])^2 + K[c,a]*A*K[c,a]*B*Pc + (Pa*A*Kr[a,a] + Kr[a,a]*A*Pa)*B*K[a,c]
    elseif a==c && c != b
        return -Pa*A*(K[a,b])^2*B*Pa + K[b,a]*A*Pb*B*K[b,a] + Pa*A*K[a,b]*B*Kr[a,a] + Kr[a,a]*A*K[a,b]*B*Pa
    elseif b==c && c != a
        return -(K[b,a])^2*A*Pb*B*Pb + Pa*A*K[a,b]*B*K[a,b] + K[b,a]*A*Pb*B*Kr[b,b] + K[b,a]*A*Kr[b,b]*B*Pb
    end
end

function elements_3_Rs(a,b,c,A,B,Ob) # regularized version
    Pa = Ob.Ps[a]
    Pb = Ob.Ps[b]
    Pc = Ob.Ps[c]
    K = Ob.Kij
    Pa*A*K[a,b]*B*K[a,c] + K[b,a]*A*Pb*B*K[b,c] + K[c,a]*A*K[c,b]*B*Pc - K[b,a]*K[c,a]*A*Pb*B*Pc - Pa*A*K[a,b]*K[c,b]*B*Pc - Pa*A*Pb*B*K[a,c]*K[b,c]
end

# Builds the first order correction of the multipoint extrapolation, the term δ_G^1 δ_g^0 δ_α^0 
function builds_order_100(Ob)
    F = zero(Ob)
    for j=1:Ob.n, a=1:Ob.n, b=1:Ob.n
        if a < b
            dG = Ob.Gij[a,b]
            A = elements_3_Rs(j,a,b,dG,dG,Ob)
            # verif = abs(norm(imag.(A))/norm(real.(A))) # verifies A is real
            # px("verif A real ",verif)
            # @assert verif < 1e-3
            F .+= Ob.αs[j]*Ob.αs[a]*Ob.αs[b]*A
        end
    end
    -Ob.Sα^(-2)*F
end

builds_order_001(Ob) = -Ob.Sα^(-2)*Ob.δα*sum(Ob.αs[i]*Ob.αs[j]*elements_2_Rs(i,j,Ob.Gs[j],Ob) for i=1:Ob.n, j=1:Ob.n)
builds_order_002(Ob) = Ob.Sα^(-3)*Ob.δα^2*sum(Ob.αs[i]*Ob.αs[j]*Ob.αs[k]*elements_3_Rs(i,j,k,Ob.Gs[j],Ob.Gs[k],Ob) for i=1:Ob.n, j=1:Ob.n, k=1:Ob.n)
builds_order_010(Ob) = Ob.Sα^(-2)*sum(Ob.αs[i]*Ob.αs[j]*elements_2_Rs(i,j,Ob.g,Ob) for i=1:Ob.n, j=1:Ob.n, k=1:Ob.n)
builds_order_020(Ob) = Ob.Sα^(-3)*sum(Ob.αs[i]*Ob.αs[j]*Ob.αs[k]*elements_3_Rs(i,j,k,Ob.g,Ob.g,Ob) for i=1:Ob.n, j=1:Ob.n, k=1:Ob.n)
builds_order_011(Ob) = -Ob.Sα^(-3)*Ob.δα*sum(Ob.αs[i]*Ob.αs[j]*Ob.αs[k]*(elements_3_Rs(i,j,k,Ob.Gs[j],Ob.g,Ob) + elements_3_Rs(i,j,k,Ob.g,Ob.Gs[k],Ob)) for i=1:Ob.n, j=1:Ob.n, k=1:Ob.n)

# Compute the objects from αs and Gs
function zero_and_first_order(Ob)
    # First and second corrections
    Ob.Fn[(1,0,0)] = builds_order_100(Ob)
    zeros = 0*Ob.Fn[(1,0,0)]
    Ob.Fn[(0,0,1)] = zeros
    Ob.Fn[(0,0,2)] = zeros
    Ob.Fn[(0,1,0)] = zeros
    Ob.Fn[(0,2,0)] = zeros
    Ob.Fn[(0,1,1)] = zeros

    condition_α = abs(Ob.δα) > 1e-10
    condition_g = abs(Ob.δg) > 1e-10

    if condition_α
        Ob.Fn[(0,0,1)] = builds_order_001(Ob)
        Ob.Fn[(0,0,2)] = builds_order_002(Ob)
    end
    if condition_g
        Ob.Fn[(0,1,0)] = builds_order_010(Ob)
        Ob.Fn[(0,2,0)] = builds_order_020(Ob)
    end
    if condition_g && condition_α
        Ob.Fn[(0,1,1)] = builds_order_011(Ob)
    end

    # Distances
    Approx0 = Ob.Fn[(0,0,0)]
    Approx1 = Approx0 + Ob.Fn[(0,0,1)] + Ob.Fn[(0,1,0)]
    dA = Ob.Fn[(0,0,2)]+Ob.Fn[(0,2,0)]+Ob.Fn[(0,1,1)]+ Ob.Fn[(1,0,0)]
    Approx2 = Approx1 + dA
    # px("Ordre 0 ",norm(Approx0)," Ordre 1 ",norm(Ob.Fn[(0,0,1)])," Ordre 2 ",norm(Ob.Fn[(0,0,2)])," δα ",Ob.δα)
    # Approx2 = Ob.Fn[(0,0,0)]+Ob.Fn[(1,0,0)]+Ob.Fn[(0,0,1)]+Ob.Fn[(0,0,2)]
    d0 = dist(Ob.F_exact,Approx0,Ob,1)
    d1 = dist(Ob.F_exact,Approx1,Ob,1)
    d2 = dist(Ob.F_exact,Approx2,Ob,1)

    # Multipoint + standard perturbation
    d3 = 0
    if Ob.with_PT
        # When with_PT is true, multipoint did the approximation to H0+αG, and we still need to go from H0+αG to H0+G
        # α = minimization_αG(Ob.G,Ob.Gs) 
        # αG = sum(α .* Ob.Gs)
        αG = Ob.α_dot_G
        g = Ob.G_with_PT - αG
        H0_PT = Ob.H0 + αG
        P, K = PT_quantities(H0_PT)
        (p1,p2,p3) = PT_series(P,g,K)
        @assert norm(dA) < 1e-10
        Approx_PT = Approx1 + p1+p2
        F_exact, E_exact = ground_projection(Ob.H0 + Ob.G_with_PT)
        d3 = dist(F_exact,Approx_PT,Ob,1)
        gp, Ep = ground_projection(Ob.H0 + αG)
        d_multi = dist(gp, Approx1,Ob,1)
        d_pt = dist(F_exact,gp + p1+p2,Ob,1)
        d3 = d_multi + d_pt
        # d3= abs(Ob.αs[2])
    end

    # Outputs the objects
    d0,d1,d2,d3
end

########### Standard perturbation theory

P1(P,g,K) = P*g*K + K*g*P
P2(P,g,K) = P*g*K*g*K + K*g*P*g*K + K*g*K*g*P - (P*g*P*g*K^2 + P*g*K^2*g*P + K^2*g*P*g*P)
P3(P,g,K) = (P*g*P*g*P*g*K^3 + P*g*P*g*K^3*g*P + P*g*K^3*g*P*g*P + K^3*g*P*g*P*g*P) + (P*g*K*g*K*g*K + K*g*P*g*K*g*K + K*g*K*g*P*g*K + K*g*K*g*K*g*P) - (P*g*P*g*K^2*g*K + P*g*P*g*K*g*K^2 + K*g*K^2*g*P*g*P + K^2*g*K*g*P*g*P + P*g*K*g*K^2*g*P + P*g*K^2*g*K*g*P + P*g*K*g*P*g*K^2 + P*g*K^2*g*P*g*K + K*g*P*g*K^2*g*P + K^2*g*P*g*K*g*P + K*g*P*g*P*g*K^2 + K^2*g*P*g*P*g*K)

function PT_quantities(H) # from H, gives PT' P and K of H
        Es,ψs = first_eigenmodes_of_matrix(H)
        M = length(Es)
        Ps = [ψs[:,k]*ψs[:,k]' for k=1:M]

        (E0,P0) = (Es[1],Ps[1])
        K0 = sum((1/(E0 - Es[k]))*Ps[k] for k=2:M)
        P0,K0
end

function PT_series(P,g,K)
    p1 = P1(P,g,K)
    p2 = P2(P,g,K)
    p3 = P3(P,g,K)
    p1,p2,p3
end

function perturbation_theory(Ob;fair=true)
    d0 = [] # zeroth orders
    d1 = [] # first orders
    d2 = [] # second orders
    d3 = [] # third orders
    for j=1:Ob.n
        g = Ob.G - Ob.Gs[j]
        K = Ob.Kij[j,j]
        P = Ob.Ps[j]
        (p1,p2,p3) = PT_series(P,g,K)
        d0p = dist(Ob.F_exact,P,Ob,1)
        d1p = dist(Ob.F_exact,P+p1,Ob,1)
        d2p = dist(Ob.F_exact,P+p1+p2,Ob,1)
        d3p = dist(Ob.F_exact,P+p1+p2+p3,Ob,1)
        push!(d0,d0p); push!(d1,d1p); push!(d2,d2p); push!(d3,d3p)
    end
    if fair
        return min(d0...), min(d1...), min(d2...), min(d3...)
    end
    j = argmin([norm(Ob.G - Gj) for Gj in Ob.Gs])
    # j = argmin(d3)
    d0[j], d1[j], d2[j], d3[j]
end

########### Test functions

##### Test of the main formula
# Complex functions

function Hz(Ob)
    function G_z(z)
        G = zero(Ob)
        for i=1:Ob.n, j=1:Ob.n
            if i < j
                G .+= (1/Ob.Sα)*Ob.αs[i]*Ob.αs[j]*Ob.Gij[i,j]*Ob.Rs[j](z)*Ob.Gij[i,j]*Ob.Rs[i](z)
            end
        end
        G
    end
    G_z
end

Az(Ob) = z -> (1/Ob.Sα)*sum(Ob.αs[j]*Ob.Gs[j]*Ob.Rs[j](z) for j=1:Ob.n)
Lz(Ob) = z -> (1/Ob.Sα)*sum(Ob.αs[j]*Ob.Rs[j](z) for j=1:Ob.n)

# L (1+ H + δα A + g L)^(-1)
function multipolar_resolvent(Ob)
    H = Hz(Ob)
    A = Az(Ob)
    L = Lz(Ob)
    z -> L(z)*inv(I*1 + H(z) + Ob.δα*A(z) - Ob.g*L(z))
end 

function test_multipolar_resolvent(Ob)
    z0 = 5+im
    rhs = multipolar_resolvent(Ob)
    lhs(z) = inv(z*I-(Ob.H0 + Ob.G))
    px("Test multipoint resolvent ",dist(rhs(z0),lhs(z0)))
end


##### Test of terms in the expansion

function first_order_element_int(a,b,c,A,B,Ob,resolution=10000) # (2πi)^(-1) ∮ Rj Gab Rb Gab Ra by complex integral
    fun(z) = Ob.Rs[a](z)*A*Ob.Rs[b](z)*B*Ob.Rs[c](z)
    E0 = Ob.Es[a]
    E1 = Ob.allEs[a][2]
    contour_integral(fun,E0,resolution,(E1-E0)/2)
end

function zeroth_order_element_int(a,b,A,Ob,resolution=10000) # (2πi)^(-1) ∮ Ra Gab Rb by complex integral
    fun(z) = Ob.Rs[a](z)*A*Ob.Rs[b](z)
    E0 = Ob.Es[a]
    E1 = Ob.allEs[a][2]
    contour_integral(fun,E0,resolution,(E1-E0)/2)
end

function test_first_order_element(Ob)
    A = Ob.H0
    px("Test zeroth order")
    for a=1:Ob.n, b=1:Ob.n
        integrated = zeroth_order_element_int(a,b,A,Ob,10000)
        analytical = elements_2_Rs(a,b,A,Ob)
        px("a b ",a," ",b," Compare analytical and integrated ",dist(analytical,integrated))
    end
    px("Test first order")
    B = Ob.Gs[1]
    for a=1:Ob.n, b=1:Ob.n, c=1:Ob.n
        integrated = first_order_element_int(a,b,c,A,B,Ob,10000)
        analytical = elements_3_Rs(a,b,c,A,B,Ob)
        if a==b && b==c
            px("Int ",norm(integrated))
            px("An ",norm(analytical))
        end
        px("a b c ",a," ",b," ",c," Compare analytical and integrated ",dist(analytical,integrated))
    end
end
