using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LaTeXStrings
using LinearAlgebra
using JLD2
using Plots

plotly()

datafilename = "mc_orbit_varied.jld2"
datapath = joinpath(@__DIR__, "..", "data", datafilename)
mc_results = load(datapath)

function mc_plot_momentum_magnitude(mc_results; max_samples=500, title="")
    sp = []
    for (controller_name, controller_data) in mc_results
        Ntrials = size(mc_results[controller_name]["T"])[1]
        for mc_step = 1:Ntrials
            xhist = mc_data[controller_name]["X"][mc_step, :, :]
            uhist = mc_data[controller_name]["U"][mc_step, :, :]
            thist = mc_data[controller_name]["T"][mc_step, 1, :]

            downsample = get_downsample(length(thist), max_samples)
            ω = xhist[11:13, downsample]
            h = 
            plot(thist[downsample] / (60 * 60), rad2deg.(xhist[11:13, downsample]'), label=["ω_x" "ω_y" "ω_z"])
            plot!(title=title, xlabel="Time (hours)", ylabel="Rates (deg/s)", linewidth=1.5)
end