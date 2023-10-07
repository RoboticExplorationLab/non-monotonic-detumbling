using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LaTeXStrings
using LinearAlgebra
using JLD2
using Plots

include("../src/satellite_simulator.jl")

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

function mc_plot_momentum_magnitude_final_histogram(mc_results, params; title="")
    Ncontrollers = length(keys(mc_results))
    J = params["satellite_model"]["inertia"]
    Ntrials = size(mc_results[collect(keys(mc_results))[1]]["T"])[1]
    h_end = zeros(Ncontrollers, Ntrials)
    controller_idx = 1
    controller_names = collect(keys(mc_results))
    for controller_name in controller_names
        for mc_step = 1:Ntrials
            xend = mc_results[controller_name]["X"][mc_step, :, end]

            ω = xend[11:13]
            h = J * ω
            h_end[controller_idx, mc_step] = norm(h)
        end
        controller_idx += 1
    end
    h_max = maximum(h_end)
    bins = range(0, h_max, Ntrials)
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

function mc_plot_detumble_time_histogram(mc_results, params; title="", terminal_threshold=0.01)
    Ncontrollers = length(keys(mc_results))
    J = params["satellite_model"]["inertia"]
    Ntrials = size(mc_results[collect(keys(mc_results))[1]]["T"])[1]
    t_done = zeros(Ncontrollers, Ntrials)
    controller_idx = 1
    controller_names = collect(keys(mc_results))
    for controller_name in controller_names
        for mc_step = 1:Ntrials
            xhist = mc_results[controller_name]["X"][mc_step, :, :]

            ω = xhist[11:13, :]
            h = J * ω
            h̄ = dropdims(sqrt.(sum(h .* h, dims=1)); dims=1)
            h_start = h̄[1]
            h_done = terminal_threshold * h_start
            h_done_idx = findfirst(h̄ .< h_done)
            if isnothing(h_done_idx)
                h_done_idx = length(h̄)
            end
            t_done[controller_idx, mc_step] = mc_results[controller_name]["T"][mc_step, 1, h_done_idx]
        end
        controller_idx += 1
    end
    t_done /= (60 * 60) # convert to hours
    t_done_max = maximum(t_done)
    bins = range(start=0.0, stop=t_done_max + (t_done_max / Ntrials), step=t_done_max / Ntrials)
    plots = []
    max_ylim = 0.0
    for i = 1:size(t_done)[1]
        p = histogram(t_done[i, :], bins=bins, title=controller_names[i], alpha=0.9, label="")
        push!(plots, p)
        max_ylim = max(max_ylim, ylims(p)[2])
    end

    plot!(plots[end], xlabel="Time (hours)")
    plot(plots...,
        plot_title=title,
        size=(500, 800),
        ylims=(0, max_ylim),
        layout=(Ncontrollers, 1),
        ylabel="Count",
        legend=false)
end

datafilename = "mc_orbit_varied_no_noise_30deg_s.jld2"
datapath = joinpath(@__DIR__, "..", "data", datafilename)
data = load(datapath)
mc_results = data["mc_results"]
params = data["params"]

display(mc_plot_momentum_magnitude_vs_time(mc_results, params; title="Momentum Magnitude"))
display(mc_plot_momentum_magnitude_final_histogram(mc_results, params; title="Final Momentum Magnitude"))
h_thresh = 0.01
display(mc_plot_detumble_time_histogram(mc_results, params; title="Detumble Time ($(h_thresh*100)% of initial)", terminal_threshold=h_thresh))
