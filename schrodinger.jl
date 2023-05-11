# include("lobpcg.jl")
include("eval_f.jl")
using FFTW, LinearAlgebra
px = println

######################## Annexe functions


######################## Misc

function axis2grid(ax,p)
    grid = []
    if p.d == 1
        return [[ax[i]] for i=1:p.N]
    elseif p.d == 2
        return [[ax[i],ax[j]] for i=1:p.N, j=1:p.N]
    elseif p.d == 3
        return [[ax[i],ax[j],ax[l]] for i=1:p.N, j=1:p.N, l=1:p.N]
    end
end

function translation_function(f,τ,p)
    if p.d == 1
        return x -> f(x .- τ)
    elseif p.d == 2
        (x,y) -> f(x .-τ[1],y .-τ[2])
    elseif p.d == 3
        (x,y,z) -> f(x .-τ[1],y .-τ[2],z .-τ[3])
    end
end

function periodizes_function(f,a,p,n=2) # f becomes a-periodic
    if p.d == 1
	g = x -> 0
	for i=-n:n
            g = g + translation_function(f,a*i,p)
        end
        return g
    elseif p.d == 2
	g = (x,y) -> 0
        for i=-n:n, j=-n:n
            g = g + translation_function(f,a*[i,j],p)
        end
        return g
    elseif p.d == 3
	g = (x,y,z) -> 0
        for i=-n:n, j=-n:n, k=-n:n
            g = g + translation_function(f,a*[i,j,k],p)
        end
        return g
    end
end

######################## Main functions

mutable struct Params
    d; N; L
    dx; x_axis; k_axis
    Nfull; dvol; x_grid
    lattice; dual_lattice 
    k_grid_red; k_grid_cart; k2_grid_cart; k2lin
    N_plots
    function Params(d,N,lattice)
        p = new(d,N)
        # lattice = [a1], [a1 a2], or [a1 a2 a3], where aj is a column vector of size d
        # dual_lattice = [a1_star], [a1_star a2_star], or [a1_star a2_star a3_star], where aj is a column vector of size d
        # aj_star = dual_lattice[1:p.d,j]
        p.lattice = d==1 ? [lattice] : lattice
        p.dual_lattice = 2π*(d==1 ? [1/lattice] : inv(p.lattice))'
        p.L = norm(lattice[1]) 

        p.dx = p.L/N
        p.x_axis = (0:N-1)*p.dx
        p.k_axis = fftfreq(N)*N
        p.x_grid = axis2grid(p.x_axis,p)

        p.Nfull = p.N^p.d
        p.dvol = p.dx^p.d
        
        # Fourier
        build_kinetic(p) # creates the grid of cartesian vectors
        p.N_plots = 100
        p
    end
end

a_star(j,p) = p.dual_lattice[1:p.d,j]

function build_kinetic(p)
    p.k_grid_red = axis2grid(p.k_axis,p)
    p.k_grid_cart = [sum(p.k_grid_red[i][j].*a_star(j,p) for j=1:p.d) for i=1:length(p.k_grid_red)]
    p.k2_grid_cart = my_zeros(p,false)
    for i=1:length(p.k_grid_cart)
        p.k2_grid_cart[i] = norm(p.k_grid_cart[i])^2
    end
    p.k2lin = vcat(p.k2_grid_cart...)
end

my_zeros(p,complex) = Base.zeros(complex ? ComplexF64 : Float64, Tuple(fill(p.N,p.d)))

integral(ϕ,p) = p.dvol*sum(ϕ)
sca(ϕ,ψ,p) = p.dvol*ϕ⋅ψ
norm2(ϕ,p) = real(sca(ϕ,ϕ,p))
norms(ϕ,p) = sqrt(norm2(ϕ,p))

cyclic_conv(a,b) = ifft(fft(a).*fft(b))

# Gives the action (hat{VX})_k in Fourier space. Vfour is the Fourier transform of V, Xfour is the Fourier transform of X
actionV(Vfour,p) = Xfour -> vcat(cyclic_conv(Vfour,Xfour)/length(Vfour)...)
actionH(actV,p) = X -> p.k2lin.*X .+ actV(Reshape(X,p)) # X = Xlin, v in direct space

Reshape(ψ,p) = reshape(ψ,Tuple(fill(p.N,p.d)))

function matrix(v_four,p)
    V = actionV(v_four,p)
    H = actionH(V,p)
    mat = zeros(ComplexF64,p.Nfull,p.Nfull)
    diag = I
    for j=1:p.Nfull
        mat[1:p.Nfull,j] = H(diag[1:p.Nfull,j])
    end
    mat
    
    # (λs,ϕs,cv) = solve_lobpcg(H,p.Nfull,1,p.k2lin;tol=1e-7)
    # ψ = Reshape(ϕs[1],p)
    # (λs[1],ψ)
end

function fun2four(f,p)
    v = eval_f(f,p.x_grid,p.d) # direct array
    v_four = fft(v) # fourier array
end

function plot_ground_state_of_matrix(mat,v_four,p;path="fig")
    sol = eigen(mat)
    ψ_four = sol.vectors[1:p.N,1]
    res_plot = 200
    plotit([ψ_four,v_four/200],p.lattice[1][1],res_plot;path=path)
end

function plot_them(vs_four,mats,p,path,colors)
    lattice = p.lattice[1][1]
    resolution = 300
    fig = CairoMakie.Figure(resolution=(600,300),fontsize=20)
    axis = CairoMakie.Axis(fig[1,1])
    xlims!(axis, -lattice/2, lattice/2)
    # Legend(fig[1,2], Ls,labels, "X quantities:", halign = :left, valign = :center)
    ls = []
    labels = [L"lala" for i=1:2*length(vs_four)]
    for i=1:length(vs_four)
        sol = eigen(mats[i])
        ψ_four = sol.vectors[1:p.N,1]
        tp = [vs_four[i],ψ_four]
        for j=1:2
            (f_dir,ax) = fourier2direct(tp[j],lattice;res=resolution,n_figs=1,cutoff=50,shifted=true)
            l = CairoMakie.lines!(ax,real.(f_dir),color=colors[i],linestyle=j==1 ? :solid : :dash)
            push!(ls,l)
            letv = raw"$V_{"*string(i)*raw"}$"
            letp = raw"$\psi_{"*string(i)*raw"}$"
            labels[2*(i-1) + j] = j==1 ? letv : letp
        end
    end
    Legend(fig[1,2], ls,labels, halign = :left, valign = :center)
    CairoMakie.save(path,fig)
end


######################## Launch

# function function2matrix(p)
    # Parameters
    # d = 1; N = 70; L = 2π
    # p = Params(d,N,L)

    # Build some function
    # some_fun(p) = x -> 2cos(-3x[1]*2π/p.L) + sin(5x[1]*2π/p.L)
    # v = eval_f(f,p.x_grid,p.d) # direct array
    # v_four = fft(v) # fourier array
    # matrix(v_four,p)

    # Solve
    # (λ,ψ_four) = solve(v_four,p)

    # Plot
    # res_plot = 100
    # px("norm ",norm(ψ_four))
    # plotit([ψ_four,v_four/500],p.d == 1 ? p.lattice[1][1] : p.lattice,res_plot)
# end
