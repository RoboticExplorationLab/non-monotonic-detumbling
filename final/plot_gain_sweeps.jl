using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LaTeXStrings
using LinearAlgebra
using JLD2
using Plots

plotly()

function gs_plot_momentum_magnitude_vs_time(gs_results, params; max_samples=500, title="")
    Ncontrollers = length(keys(gs_results))
    plots = []
    J = params["satellite_model"]["inertia"]
    max_ylim = 0.0
    for (controller_name, _) in gs_results
        Ntrials = size(gs_results[controller_name]["T"])[1]
        p = plot()
        gains = gs_results[controller_name]["gains"]
        for gain_idx in eachindex(gains)
            gain = gains[gain_idx]
            xhist = gs_results[controller_name]["X"][gain_idx, :, :]
            uhist = gs_results[controller_name]["U"][gain_idx, :, :]
            thist = gs_results[controller_name]["T"][gain_idx, 1, :]

            downsample = get_downsample(length(thist), max_samples)
            ω = xhist[11:13, downsample]
            h = J * ω
            h̄ = dropdims(sqrt.(sum(h .* h, dims=1)); dims=1)
            plot!(p, thist[downsample] / (60 * 60), h̄, title=controller_name, bottom_margin=(2.0, :mm), label="k = $gain")
        end
        max_ylim = max(max_ylim, ylims(p)[2])
        plot!(p, legend=true)
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

datafilename = "gain_sweep_all.jld2"
datapath = joinpath(@__DIR__, "..", "data", datafilename)
data = load(datapath)
gs_results = data["gs_results"]
params = data["params"]

display(gs_plot_momentum_magnitude_vs_time(gs_results, params; title="Gain Sweep Momentum Magnitude"))