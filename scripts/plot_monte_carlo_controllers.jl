using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LaTeXStrings
using LinearAlgebra
using JLD2
using Plots

plotly()

function mc_plot_momentum_magnitude_vs_time(mc_results, params; max_samples=500, title="")
    Ncontrollers = length(keys(mc_results))
    plots = []
    J = params["satellite_model"]["inertia"]
    max_ylim = 0.0
    for (controller_name, controller_data) in mc_results
        Ntrials = size(mc_results[controller_name]["T"])[1]
        p = plot()
        for mc_step = 1:Ntrials
            xhist = mc_results[controller_name]["X"][mc_step, :, :]
            uhist = mc_results[controller_name]["U"][mc_step, :, :]
            thist = mc_results[controller_name]["T"][mc_step, 1, :]

            downsample = get_downsample(length(thist), max_samples)
            ω = xhist[11:13, downsample]
            h = J * ω
            h̄ = dropdims(sqrt.(sum(h .* h, dims=1)); dims=1)
            plot!(p, thist[downsample] / (60 * 60), h̄, title=controller_name, bottom_margin=(2.0, :mm))
        end
        max_ylim = max(max_ylim, ylims(p)[2])
        plot!(p, legend=false)
        push!(plots, p)
    end
    plot!(plots[end], xlabel="Time (hours)")
    plot(plots...,
        plot_title=title,
        layout=(Ncontrollers, 1),
        size=(600, 800),
        ylims=(0, max_ylim),
        ylabel="Momentum (Nms)",
        linewidth=1.5,
        legend=false)
end

function mc_plot_momentum_magnitude_final_histogram(mc_results, params; max_samples=500, title="")
    Ncontrollers = length(keys(mc_results))
    J = params["satellite_model"]["inertia"]
    h_end = zeros(Ncontrollers, Ntrials)
    controller_idx = 1
    controller_names = collect(keys(mc_results))
    for controller_name in controller_names
        Ntrials = size(mc_results[controller_name]["T"])[1]
        for mc_step = 1:Ntrials
            xend = mc_results[controller_name]["X"][mc_step, :, end]

            ω = xend[11:13]
            h = J * ω
            h_end[controller_idx, mc_step] = norm(h)
        end
        controller_idx += 1
    end
    h_max = maximum(h_end)
    bins = range(0, h_max, 100)
    plots = []
    max_ylim = 0.0
    for i = 1:size(h_end)[1]
        p = histogram(h_end[i, :], bins=bins, title=controller_names[i], alpha=0.9, label="")
        push!(plots, p)
        max_ylim = max(max_ylim, ylims(p)[2])
    end

    plot!(plots[end], xlabel="Momentum (Nms)")
    plot(plots...,
        plot_title=title,
        size=(500, 800),
        ylims=(0, max_ylim),
        layout=(Ncontrollers, 1),
        ylabel="Count",
        legend=false)
end

datafilename = "mc_orbit_varied.jld2"
datapath = joinpath(@__DIR__, "..", "data", datafilename)
data = load(datapath)
mc_results = data["mc_results"]
params = data["params"]

display(mc_plot_momentum_magnitude_vs_time(mc_results, params; title="Momentum Magnitude"))
display(mc_plot_momentum_magnitude_final_histogram(mc_results, params; title="Momentum Magnitude"))