# This code applies the functions of functions_multipoint_perturbation.jl to apply multipoint perturbation to Schrödinger operators
using CairoMakie, LaTeXStrings
include("functions_multipoint_perturbation.jl")
include("schrodinger.jl")
output_path = "figs/"

function δG_to_zero(H0,G1,A,B,C,αs,name;label_y=true)
    N = 4 # number of Gs
    εs = 10 .^((-3:0.05:1))
    rel_dist = true

    F0s = []; F1s = []; F2s = []; dGs = []
    d0ps = []; d1ps = []; d2ps = []; d3ps = []
    Ob = ""
    for a=1:length(εs)
        G2 = G1 + A*(εs[a])
        G3 = G1 + B*(εs[a])
        G4 = G1 + C*(εs[a])
        Gs = [G1,G2,G3,G4][1:N]

        # Solves and builds necessary objects
        Ob = Objects(H0,Gs,rel_dist;αs=αs[1:N])

        # Tests for complex integration
        # test_first_order_element(Ob)
        # test_multipolar_resolvent(Ob)

        d0,d1,d2,d3 = zero_and_first_order(Ob)

        # Perturbation theory
        d0p, d1p, d2p, d3p = perturbation_theory(Ob)

        push!(F0s,d0); push!(F1s,d1); push!(F2s,d2); push!(dGs,Ob.δG)
        push!(d0ps,d0p); push!(d1ps,d1p); push!(d2ps,d2p); push!(d3ps,d3p)
    end

    size_txt = 20
    fig = Figure(resolution=(300, 400), fontsize=size_txt)
    ax = Axis(fig[1, 1], title="", xlabel = L"\epsilon", ylabel = label_y ? L"D_e(\bscrF_{\text{exact}} \, , \, \bscrF_{\text{approx}})" : "", xscale=log10,yscale=log10)
    xlims!(ax, minimum(εs), maximum(εs))
    ax.xreversed = true
    points(Fs) = [Makie.Point2f0(εs[i],Fs[i]) for i=1:length(εs)]

    cols = [:red,:green,:cyan,:orange,:blue,:brown,:black]
    toplot = [F0s,F2s,d0ps,d1ps,d2ps,d3ps]
    labels = [#L"\delta_{G}",
              L"\mathbb{D}_{0} ",
              L"\mathbb{D}_{2}",
              L"\mathbb{P}_{0} ",
              L"\mathbb{P}_{1} ",
              L"\mathbb{P}_{2} ",
              L"\mathbb{P}_{3} "
             ]
    Ls = []
    for j=1:length(toplot)
        l = lines!(points(toplot[j]),color=cols[j])
        push!(Ls,l)
    end
    # Annotations
    if abs(sum(αs)-1)<1e-5
        slopes = [-2,-4,-1,-2,-3,-4]
        xs_ys = [(1.5,4),(1.5,9),(1.5,2),(1.5,6),(1.5,8),(1.5,11),]
        for j=1:length(toplot)
            x,y = xs_ys[j]
            text!(10.0^(-x),10.0^(-y) , text = "slope "*string(slopes[j]), color = cols[j])#, align = (:center, :center))
        end
    end
    save(Ob.path_plots*"compare2pert_thy"*name*".pdf",fig)

    # Legend
    fig_leg = Figure(resolution=(300, 400), fontsize=size_txt)
    Legend(fig_leg[1,2], Ls,labels, L"\bscrF_{\text{approx}} \text{ quantities}:", halign = :left, valign = :center)
    save(Ob.path_plots*"compare2pert_thy_legend.pdf",fig_leg)
end

function δα_heatmap(H0,G1,G2,range_α;log=true,fig_ttl="",expensive=false,fair=true,reso=-1)
    Gs = [G1,G2]
    center_α_x, center_α_y = range_α[1]
    width_α = range_α[2]
    αmin_x = center_α_x-width_α
    αmax_x = center_α_x+width_α

    αmin_y = center_α_y-width_α
    αmax_y = center_α_y+width_α
    res = expensive ? 150 : 50
    if reso!= -1
        res = reso
    end
    rel_dist = true
    dx = (αmax_x-αmin_x)/(res)
    axes_x = (αmin_x:dx:αmax_x)
    axes_y = (αmin_y:dx:αmax_y)
    N = length(axes_x)
    grid = [[α1,α2] for α1 in axes_x, α2 in axes_y]
    n_plots = 6
    errors = [[0.0 for α1 in axes_x, α2 in axes_y] for j=1:n_plots]
    Ob = ""

    for p=1:length(grid)
        αs = grid[p]

        # Solves and builds necessary objects
        Ob = Objects(H0,Gs,rel_dist;αs=αs)

        d0,d1,d2,d3 = zero_and_first_order(Ob)
        d0p, d1p, d2p, d3p = perturbation_theory(Ob,fair=fair)

        ds = [d0p,d1p,d2p,d0,d1,d2]

        for i=1:n_plots
            errors[i][p] = log ? log10(ds[i]) : ds[i]
        end
    end

    # Plots errors
    mini = minimum(minimum.(errors[1:n_plots]))
    maxi = maximum(maximum.(errors[1:n_plots]))
    clims = (mini,maxi)

    size_txt = 30
    fig = Figure(resolution=(400*n_plots,500), fontsize=size_txt)
    ga = fig[1, 1] = GridLayout()
    colgap!(ga, 0)
    rowgap!(ga, 0)
    rowgap!(fig.layout, 0)
    colgap!(fig.layout, 0)
    hm = ""

    for m=1:n_plots
        ax, hm = heatmap(ga[1,m], axes_x, axes_y, errors[m]; colorrange=clims, colormap=:Spectral)#, axis=(;cscale=log10))
        ax.aspect = 1
        ax.xlabelsize=size_txt
        ax.ylabelsize=size_txt
        if m >= 2
            hidedecorations!(ax, grid = false)
        end
        xs = αmin_x:dx:αmax_x
        ys = 1 .- xs
        lines!(xs, ys, linewidth = 2, color = :black, linestyle=:dash)
        # Plots points
        pp = [[0,0], [0,1], [1,0]]
        for pt in pp
            X = pt[1]; Y = pt[2]
            if αmin_x <= X <= αmax_x && αmin_y <= Y <= αmax_y
                scatter!(ax, [X], [Y]; markersize=10, color = :black)
            end
        end
    end
    Colorbar(ga[1,n_plots+1], hm)
    save(Ob.path_plots*fig_ttl*"_delta_alpha"*(fair ? "" : "_unfair")*".pdf",fig)
end

function δα_lineplot(H0,G1,G2)
    Gs = [G1,G2]
    εmin_log = -5
    εmax_log = 0.5
    res = 150
    rel_dist = true
    dx = (εmax_log-εmin_log)/(res)
    εs = 10 .^((εmin_log:dx:εmax_log))
    αs = [[1/2 * (1 + ε), 1/2 * (1 + ε)] for ε in εs]
    N = length(εs)
    Ob = ""
    n_plots = 6
    errors = [copy(zeros(N)) for i=1:n_plots]
    for p=1:N
        G = sum(αs[p] .*Gs)
        # Solves and builds necessary objects
        Ob = Objects(H0,Gs,rel_dist;αs=αs[p])
        d0,d1,d2,d3 = zero_and_first_order(Ob)
        d0p, d1p, d2p, d3p = perturbation_theory(Ob)
        ds = [d0,d1,d2,d0p,d1p,d2p]
        for i=1:n_plots
            errors[i][p] = ds[i]
        end
    end

    # Plots errors
    size_txt = 15
    fig = Figure(resolution=(600, 300), fontsize=size_txt)
    ax = Axis(fig[1, 1], title="", xlabel = L"\epsilon", ylabel = L"D_e(\bscrF_{\text{exact}} \, , \, \bscrF_{\text{approx}})", xscale=log10,yscale=log10)
    xlims!(ax, minimum(εs), maximum(εs))
    ax.xreversed = true

    cols = [:red,:green,:cyan,:orange,:blue,:brown]
    labels = [
              L"\mathbb{D}_{0} ",
              L"\mathbb{D}_{1} ",
              L"\mathbb{D}_{2}",
              L"\mathbb{P}_{0} ",
              L"\mathbb{P}_{1} ",
              L"\mathbb{P}_{2} "
             ]
    Ls = []
    points(Fs) = [Makie.Point2f0(εs[i],Fs[i]) for i=1:N]
    for j=1:n_plots
        l = lines!(points(errors[j]),color=cols[j])
        push!(Ls,l)
    end
    Legend(fig[1,2], Ls, labels[1:n_plots], L"\bscrF_{\text{approx}} \text{ quantities}:", halign = :left, valign = :center)
    save(Ob.path_plots*"errors_on_errors_delta_alpha.pdf",fig)
end

function delta_g_lineplot(H0,G1,G2,G3,αs,name;label_y=true)
    εmin_log = -7
    εmax_log = 0
    res = 100
    dx = (εmax_log-εmin_log)/(res)
    εs = 10 .^((εmin_log:dx:εmax_log))
    Gs = [G1,G2]
    Ob = ""
    rel_dist = true
    n_error_funs = 3
    n_minimization_methods = 3
    mins_methods = [1,0,"infinity"]
    N = length(εs)
    err_multipert = [copy(zeros(N)) for i=1:n_error_funs,j=1:n_minimization_methods]
    err_standard_pert = [copy(zeros(N)) for i=1:3]
    sum_αG = sum(αs .*Gs)
    for p=1:N
        G = sum_αG + εs[p]*G3

        # Solves and builds necessary objects
        for j=1:n_minimization_methods
            Ob = Objects(H0,Gs,rel_dist;G=G,coef_δα_minimization=mins_methods[j])

            d0,d1,d2,d3 = zero_and_first_order(Ob)
            d0p, d1p, d2p, d3p = perturbation_theory(Ob)
            ds = [d0,d1,d2]
            ds_pt = [d0p, d1p, d2p]

            for i=1:n_error_funs
                err_multipert[i,j][p] = ds[i]
            end
            for i=1:3
                err_standard_pert[i][p] = ds_pt[i]
            end
        end
    end

    # Plots errors
    size_txt = 15
    fig = Figure(resolution=(300, 500), fontsize=size_txt)
    ax = Axis(fig[1, 1], title="", xlabel = L"\epsilon", ylabel = label_y ? L"D_e(\bscrF_{\text{exact}} \, , \, \bscrF_{\text{approx}})" : "", xscale=log10,yscale=log10)
    xlims!(ax, minimum(εs), maximum(εs))
    ax.xreversed = true

    cols = [:red,:green,:cyan,:orange,:blue,:brown]
    styles = [:solid,:dash,:dot]
    Ls = []
    mins_methods_str = ["1","0","+∞"]
    points(Fs) = [Makie.Point2f0(εs[i],Fs[i]) for i=1:N]
    labels_tot = [L"lala" for i=1:n_error_funs*(n_minimization_methods+1)]
    i_lab_tot = 1
    labels_k = [
              L"\mathbb{D}_{0} ",
              L"\mathbb{D}_{1} ",
              L"\mathbb{D}_{2}",
             ]

    # Pushes multipoint perturbation
    for j=1:n_minimization_methods, i=1:n_error_funs
        l = lines!(points(err_multipert[i,j]),color=cols[j],linestyle=styles[i])
        push!(Ls,l)

        txt = string(L" "*labels_k[i]*L", \, \xi = "*mins_methods_str[j])
        labels_tot[i_lab_tot] = txt
        i_lab_tot += 1
    end

    labels_pt = [
              L"\mathbb{P}_{0} ",
              L"\mathbb{P}_{1} ",
              L"\mathbb{P}_{2}",
              ]
    # Pushes standard perturbation
    for i=1:3
        l = lines!(points(err_standard_pert[i]),color=cols[i+n_error_funs])
        push!(Ls,l)
        labels_tot[i_lab_tot] = labels_pt[i]
        i_lab_tot += 1
    end
    save(Ob.path_plots*"min_methods"*name*".pdf",fig)

    # Legend
    fig_leg = Figure(resolution=(300, 400), fontsize=size_txt)
    Legend(fig_leg[1,2], Ls, labels_tot, L"\bscrF_{\text{approx}}, ξ \text{ quantities}:")
    save(Ob.path_plots*"min_methods_legend.pdf",fig_leg)
end



function convergence_basis_N()
    Ns = (2:1:100)
    M = length(Ns)
    rel_dist = true
    n_plots = 1
    errors = [[] for i=1:M]
    Ob = ""
    for j=1:M
        (H0,G0,G1,G2,G3) = create_schrodinger_potentials(true;Ni=Ns[j])
        Gs = [G1,G2]
        αs = [1/2,1/2]
        Ob = Objects(H0,Gs,rel_dist;αs=αs)
        d0,d1,d2,d3 = zero_and_first_order(Ob)
        d0p, d1p, d2p, d3p = perturbation_theory(Ob)
        ds = [Ob.E_exact]#, Ob.F_exact]
        errors[j] = ds
    end

    for i=1:M
        errors[i][1] = abs(errors[i][1]-errors[end][1])
        # errors[i][2] = abs(errors[i][2]-errors[end][2])
    end

    size_txt = 20
    fig = Figure(resolution=(700, 400), fontsize=size_txt)
    ax = Axis(fig[1, 1],title="",xlabel = L"M",ylabel = L"E(M)",yscale=log10)
    xlims!(ax, minimum(Ns), maximum(Ns[1:M-1]))
    colors = [:red,:green,:cyan,:orange,:blue,:brown]
    Ls = []
    points(Fs) = [Makie.Point2f0(Ns[i],Fs[i]) for i=1:M-1]
    labels = [
              L"D_e(\bscrF(G)(M) - \bscrF(G)(M=100))",
              L"D_e",
             ]

    for i=1:n_plots
        e = [errors[j][i] for j=1:M-1]
        l = lines!(points(e),color=colors[i])
        push!(Ls,l)
    end
    # Legend(fig[1,2], Ls, labels, "X, quantities:")#, halign = :left, valign = :center)
    save(Ob.path_plots*"convergence_quantities.pdf",fig)
end


function n_increase()
    ns = (2:5:30)
    rel_dist = true
    n_plots = 6
    Ob = ""
    H0, Gs, p = create_n_schrodinger_potentials(maximum(ns))
    M = length(Gs)
    @assert maximum(ns) == M
    errors = [zeros(n_plots) for i=1:M]

    G = sum(Gs)/M
    αs = [1/M for i=1:M]
    px("S= ",sum(αs))
    for j=1:length(ns)
        n = ns[j]
        Gsj = Gs[1:n]
        αsj = αs[1:n]
        Ob = Objects(H0,Gsj,rel_dist;G=G)
        # if j==length(ns)
            # Ob = Objects(H0,Gsj,rel_dist;αs=αs)
        # end
        px("diff ",dist(G , sum(αsj .*Gsj)))
        d0,d1,d2,d3 = zero_and_first_order(Ob)
        d0p, d1p, d2p, d3p = perturbation_theory(Ob;fair=false)
        ds = [d0,d1,d2,d0p,d1p,d2p]
        errors[j] = ds
    end
    px(errors)

    size_txt = 20
    fig = Figure(resolution=(700, 400), fontsize=size_txt)
    ax = Axis(fig[1, 1],title="",xlabel = L"M",ylabel = L"D_a(N,X\bscrF_{\text{approx}})",yscale=log10)
    xlims!(ax, minimum(ns), maximum(ns))
    colors = [:red,:green,:cyan,:orange,:blue,:brown]
    Ls = []
    points(Fs) = [Makie.Point2f0(ns[i],Fs[i]) for i=1:length(ns)]
    labels = [
              L"\mathbb{D}_{0} ",
              L"\mathbb{D}_{1} ",
              L"\mathbb{D}_{2} ",
              L"\mathbb{P}_{0} ",
              L"\mathbb{P}_{1}",
              L"\mathbb{P}_{2} ",
             ]
    for i=1:n_plots
        e = [errors[j][i] for j=1:length(ns)]
        l = lines!(points(e),color=colors[i])
        push!(Ls,l)
    end
    Legend(fig[1,2], Ls, labels, "X quantities:")#, halign = :left, valign = :center)
    save(Ob.path_plots*"n_increase.pdf",fig)
end

function apply_compare_rand_matrices()
    N = 5
    H0 = rand_herm(N)
    G1 = rand_herm(N)
    A = rand_herm(N)
    B = rand_herm(N)
    C = rand_herm(N)
    δG_to_zero(H0,G1,A,B,C)
end

function create_schrodinger_potentials(ex;Ni=-1,fact=-1,plot=false,fact_list=-1)
    N = ex ? 30 : 10
    if Ni != -1
        N = Ni
    end
    if plot
        N = 50
    end
    d = 1
    L = 2π
    p = Params(d,N,L)

    H0 = matrix(zeros(Float64,p.N),p)

    # Define potentials
    V0 = x -> -5*exp(-(x[1])^2/0.05)
    V1 = x -> -5*exp(-(x[1] - p.L/4)^2/0.05)-6*exp(-(x[1] + p.L/4)^2/0.2)
    V2 = x -> cos(-7x[1]*2π/p.L) + cos(-2x[1]*2π/p.L)
    V3 = x -> -cos(x[1]*2π/p.L) + sin(4x[1]*2π/p.L)

    G0,v0 = fun2mat_v(V0,p)
    G1,v1 = fun2mat_v(V1,p)
    G2,v2 = fun2mat_v(V2,p)
    G3,v3 = fun2mat_v(V3,p)

    # Build matrices
    if fact != -1 && fact_list==-1
        G0 = G0
        G1 = G0 + G1/fact
        G2 = G0 + G2/fact
        G3 = G0 + G3/fact
    end
    if fact_list!=-1
        G0 = G0
        G1 = G0 + G1/fact_list[1]
        G2 = G0 + G2/fact_list[2]
        G3 = G0 + G3/fact_list[3]
    end

    if plot
        colors = [:blue,:red,:black,:orange]
        vs = [v0,v1,v2,v3]/200
        mats = [G0,G1,G2,G3]
        plot_them(vs,mats,p,output_path*"pots.pdf",colors)
    end

    # Computations
    (H0,G0,G1,G2,G3)
end

plot_v_and_ψ() = create_schrodinger_potentials(true;plot=true)

function fun2mat_v(f,p)
    V = periodizes_function(f,p.L,p)
    v = fun2four(V,p)
    matrix(v,p),v
end

function create_n_schrodinger_potentials(n)
    N = 35
    d = 1
    L = 2π
    p = Params(d,N,L)

    H0 = matrix(zeros(Float64,p.N),p)
    Gs = []
    function add(f)
        if n > length(Gs)
            G,_ = fun2mat_v(f,p)
            push!(Gs,G)
        end
    end
    add(x -> 0)
    for j=1:999
        f = x -> cos(j*x[1]*2π/p.L)
        add(f)
        g = x -> sin(j*x[1]*2π/p.L)
        add(g)
        if j >= n
            break
        end
    end
    H0,Gs,p
end

function apply_compare_schrodinger(ex)
    (H0,G0,G1,G2,G3) = create_schrodinger_potentials(ex)
    αs = [-0.4,0.5,0.4,0.7]; name = ""
    δG_to_zero(H0,G0,G1,G2,G3,αs,name;label_y=false)
    αs = [-0.3,0.4,0.3,0.6]; name = "_delta_alpha_zero"
    δG_to_zero(H0,G0,G1,G2,G3,αs,name;label_y=true)
end

function plot_heatmaps(ex;reso=50)
    (H0,G0,G1,G2,G3) = create_schrodinger_potentials(ex;fact=5)

    function make(fair)
        δα_heatmap(H0,G0,G1,[(-2,3),0.4];fig_ttl="very_close_not_centered",log=true,expensive=ex,fair=fair,reso=reso)
        δα_heatmap(H0,G0,G1,[(1/2,1/2),0.2];fig_ttl="very_close",log=true,expensive=ex,fair=fair,reso=reso)
        δα_heatmap(H0,G0,G1,[(1/2,1/2),0.6];fig_ttl="close",log=true,expensive=ex,fair=fair,reso=reso)
        δα_heatmap(H0,G0,G1,[(1/2,1/2),4];fig_ttl="far",log=false,expensive=ex,fair=fair,reso=reso)
        δα_heatmap(H0,G0,G1,[(1/2,1/2),50];fig_ttl="very_far",log=false,expensive=ex,fair=fair,reso=reso)
    end
    make(true)
    # make(false)
    δα_lineplot(H0,G0,G1)
end

function delta_g_to_zero(ex)
    (H0,G0,G1,G2,G3) = create_schrodinger_potentials(ex;fact=5)#,fact_list=[5,5,1])

    αs = [0.3,1.2]; name = ""
    delta_g_lineplot(H0,G1,G2,G3,αs,name)

    αs = [0.3,0.7]; name = "_delta_alpha_zero"
    delta_g_lineplot(H0,G1,G2,G3,αs,name;label_y=false)
end

############## RUN

expensive = true # whether we do expensive computations, to have converged (in the number of planewaves used to discretiez the Hilbert's space) plots, or whether we just want to make rapid tests

# Schrödinger
apply_compare_schrodinger(expensive)
plot_heatmaps(expensive;reso=150)
delta_g_to_zero(expensive)
plot_v_and_ψ()
# convergence_basis_N()
### n_increase() # does not show that MPT is good
