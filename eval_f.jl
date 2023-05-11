# Transforms arrays (functions) in the Fourier cube to functions in cartesian space
using LinearAlgebra, CairoMakie, FFTW
import Base.+  
+(f::Function, g::Function) = (x...) -> f(x...) + g(x...) 
px = println

function scale_fun(f,λ,dim)
	if dim==1
		return x -> f(λ*x)
	elseif dim==2
		return (x,y) -> f(λ*x,λ*y)
	elseif dim==3
		return (x,y,z) -> f(λ*x,λ*y,λ*z)
	end
end

# Evaluates f with resolution res
function eval_f(f,ax,d)
    if d==1
        return f.(ax)
    elseif d==2
        return [f(ax[1][i],ax[2][j]) for i=1:length(ax[1]), j=1:length(ax[2])]
    elseif d==3
        return [f(ax[1][i],ax[2][j],ax[3][l]) for i=1:length(ax[1]), j=1:length(ax[2]), l=1:length(ax[3])]
    end
end

function get_dual_vectors(lattice)
    d = get_dim(lattice)
    if d==1
        return 2π/lattice
    elseif d==2
        dual_lat = 2π*inv([lattice[1] lattice[2]]')
        return [dual_lat[1:d,1],dual_lat[1:d,2]]
    elseif d==3
        dual_lat = 2π*inv([lattice[1] lattice[2] lattice[3]]')
        return [dual_lat[1:d,1],dual_lat[1:d,2],dual_lat[1:d,3]]
    end
end

function dims_vec(ψ)
    dim = length(size(ψ))
    lens = []; axs_k = []
    for d=1:dim
        push!(lens,size(ψ,d))
        push!(axs_k,FFTW.fftfreq(lens[d])*lens[d])
    end
    (dim,lens,axs_k)
end

function get_vol(lattice)
    d = get_dim(lattice)
    vol = 0
    if d==1
        vol = lattice
    elseif d==2
        vol = det([lattice[1] lattice[2]])
    elseif d==3
        vol = det([lattice[1] lattice[2] lattice[3]])
    end
    abs(vol)
end

function get_dim(lattice)
    if typeof(lattice) in [Float64,Int64]
        return 1
    else
        return length(lattice)
    end
end

function get_lengths(lattice)
    d = get_dim(lattice)
    if d==1
        return lattice
    else
        return norm.(lattice)
    end
end

function red_arr2cart_fun(ψ_four,lattice;k_shift0=-1,cutoff=4) # lattice = L in d=1 or [a1;a2] in d=2 or [a1;a2;a3] in d=3
    (dim,lens,axs) = dims_vec(ψ_four)
    @assert get_dim(lattice)==dim
    K = k_shift0==-1 ? zeros(dim) : k_shift0
    vol = get_vol(lattice)
    dual_vecs = get_dual_vectors(lattice)
    if dim==1
        a = x -> 0
        for i=1:lens[1]
            ki = axs[1][i]
            if abs(ki) <= cutoff
                ma_star = (ki+K[1])*dual_vecs
                c(x) = ψ_four[i]*cis(ma_star*x)/sqrt(vol)
                a = a + c
            end
        end
        return a
    elseif dim==2
        a(x,y) = 0
        for i=1:lens[1]
            ki = axs[1][i]
            if abs(ki) <= cutoff
                for j=1:lens[2]
                    kj = axs[2][j]
                    if abs(kj) <= cutoff
                        px("dual ",dual_vecs[1])
                        ma_star = (ki+K[1])*dual_vecs[1]+(kj+K[2])*dual_vecs[2]
                        c = (x,y) -> ψ_four[i,j]*cis(ma_star⋅[x,y])/sqrt(vol)
                        a = a + c
                    end
                end
            end
        end
        return a
    elseif dim==3
        a(x,y,z) = 0
        for i=1:lens[1]
            ki = axs[1][i]
            if abs(ki) <= cutoff
                for j=1:lens[2]
                    kj = axs[2][j]
                    if abs(kj) <= cutoff
                        for l=1:lens[3]
                            kl = axs[3][l]
                            if abs(kl) <= cutoff
                                ma_star = (ki+K[1])*dual_vecs[1]+(kj+K[2])*dual_vecs[2]+(kl+K[3])*dual_vecs[3]
                                c(x,y,z) = ψ_four[i,j,l]*cis(ma_star⋅[x,y,z])/sqrt(vol)
                                a = a + c
                            end
                        end
                    end
                end
            end
        end
        return a
    end
end

function fourier2direct(ψ_four,lattice;k_shift=-1,cutoff=3,res=10,n_figs=1,shifted=false) # shifted : if we translate 1/2 of the total lengths
    d = get_dim(lattice)
    f = red_arr2cart_fun(ψ_four,lattice;k_shift0=k_shift,cutoff=cutoff)
    axs = direct_axis(lattice,n_figs,res;shifted=shifted)
    (eval_f(f,axs,d),axs)
end

function one_axis(L,res,shifted=false)
    starts = shifted ? -L/2 : 0
    ends = shifted ? L/2 : L
    [starts + (i-1)*(ends-starts)/res for i=1:res]
end

function direct_axis(lattice,n_figs,res;shifted=false)
    d = get_dim(lattice)
    lens = get_lengths(lattice)
    if d==1
        return n_figs*one_axis(lens,res,shifted)
    elseif d==2
        return [n_figs*one_axis(lens[1],res,shifted),n_figs*one_axis(lens[2],res,shifted)]
    elseif d==3
        return [n_figs*one_axis(lens[1],res,shifted),n_figs*one_axis(lens[2],res,shifted),n_figs*one_axis(lens[3],res,shifted)]
    end
end

function periodize(f,n)
    g(x) = 0
    for i=-n:n
        h(x) = f(x-i)
        g = g + h
    end
    g
end

function plotit(fs_four,lattice,res_plot,n_figs=1;path="example_plot1d")
    fig = CairoMakie.Figure(resolution=(1000,1000))
    CairoMakie.Axis(fig[1,1])
    for i=1:length(fs_four)
        (f_dir,ax) = fourier2direct(fs_four[i],lattice;res=res_plot,n_figs=n_figs,cutoff=50,shifted=true)
        CairoMakie.lines!(ax,real.(f_dir))
    end
    CairoMakie.save(path*".pdf",fig)
end

function example_1d()
    # Create an example function
    a = 0.5; b = 0.2
    σ = 0.1
    f(x) = exp(-norm(x-a)^2/(2*σ^2)) + 0.3*exp(-norm(x-b)^2/(2*σ^2))
    g = periodize(f,3)
    L = 3
    lattice = L
    res = 100
    ψ = [g(i/res) for i=0:res-1]
    ψ_four = FFTW.fft(ψ)
    
    # Plot
    res_plot = 300
    n_figs = 2
    plotit(ψ_four,lattice,res_plot,n_figs)
end

function example_2d()
    # Create an example function
    a = [0.5,0.5]
    b = [0.3,0.7]
    σ = 0.1
    f(x,y) = exp(-norm([x,y]-a)^2/(2*σ^2)) + 0.3*exp(-norm([x,y]-b)^2/(2*σ^2))
    a1 = [1;0.1]
    a2 = [0.2;1]
    lattice = [a1,a2]
    res = 300
    ψ = [f(i/res,j/res) for i=0:res-1, j=0:res-1]
    ψ_four = FFTW.fft(ψ)
    
    # Plot
    res_plot = 100
    n_figs = 2
    (f_dir,ax) = fourier2direct(ψ_four,lattice;res=res_plot,n_figs=n_figs,cutoff=4,shifted=true)

    fig = CairoMakie.Figure(resolution=(1000,1000))
    CairoMakie.heatmap(fig[1,1],ax[1],ax[2],real.(f_dir))
    CairoMakie.save("example_plot2d.pdf",fig)
end

# example_1d()
# example_2d()
