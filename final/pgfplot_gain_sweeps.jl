using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LaTeXStrings
using LinearAlgebra
using JLD2
using Colors
using PGFPlotsX
using Formatting

include("../src/satellite_simulator.jl")

SAVEAS_PDF = true

function pgf_gs_plot_momentum_magnitude_vs_time(gs_results, params; max_samples=500, file_suffix="")
    Ncontrollers = length(keys(gs_results))
    plots = []
    J = params["satellite_model"]["inertia"]
    max_ylim = 0.0
    for (controller_name, _) in gs_results
        Ntrials = size(gs_results[controller_name]["T"])[1]
        color_list = distinguishable_colors(Ntrials, [RGB(1, 1, 1), RGB(0, 0, 0)], dropseed=true)
        plot_incs = []
        legend_entries = []
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
            t_plot = thist[downsample] / (60 * 60)
            if gain_idx == 3
                lineopts = @pgf {no_marks, line_width = "2pt", style = "solid", color = color_list[gain_idx], opacity = 0.8, on_layer = "axis descriptions"}
            else
                lineopts = @pgf {no_marks, line_width = "1.25pt", style = "densely dotted", color = color_list[gain_idx], opacity = 0.8}
            end
            push!(plot_incs,
                PlotInc(lineopts, Coordinates(t_plot, h̄)),
            )
            push!(
                legend_entries,
                controller_name == "Projection-based" ? format("\$k_1\$ = {:.2e}", gain) : format("k = {:.2e}", gain)
            )
            max_ylim = max(max_ylim, maximum(h̄))
        end
        legend_style = @pgf {
            font = raw"\footnotesize",
        }
        p = @pgf Axis(
            {
                set_layers = "standard",
                xmajorgrids,
                ymajorgrids,
                xlabel = "Time (hours)",
                ylabel = L"$\|h\|$ (Nms)",
                legend_pos = "north east",
                title = controller_name,
                legend_style = legend_style,
            },
            plot_incs...,
            Legend(legend_entries)
        )
        push!(plots, p)
    end
    @pgf groupopts = {
        group_style = {group_size = "3 by 2", horizontal_sep = "0.5in", vertical_sep = "0.9in"},
        height = "2in",
        width = "3.5in",
        ymin = 0,
        ymax = 1.1 * max_ylim
    }
    @pgf gp = GroupPlot(groupopts, plots...)

    pgfsave(joinpath(@__DIR__, "..", "figs", "pdf", "gain_sweep_momentum_magnitude_vs_time" * file_suffix * ".pdf"), gp)
    pgfsave(joinpath(@__DIR__, "..", "figs", "gain_sweep_momentum_magnitude_vs_time" * file_suffix * ".tikz"), gp, include_preamble=false)
    return gp
end

function pgf_gs_plot_final_momentum_vs_gain(gs_results, params; file_suffix="")
    Ncontrollers = length(keys(gs_results))
    color_list = distinguishable_colors(Ncontrollers, [RGB(1, 1, 1), RGB(0, 0, 0)], dropseed=true)
    plots = []
    J = params["satellite_model"]["inertia"]
    max_ylim = 0.0
    min_ylim = Inf
    controller_idx = 1
    for (controller_name, _) in gs_results
        Ntrials = size(gs_results[controller_name]["T"])[1]
        color_list = distinguishable_colors(Ntrials, [RGB(1, 1, 1), RGB(0, 0, 0)], dropseed=true)
        gains = gs_results[controller_name]["gains"]
        plot_incs = []
        for gain_idx in eachindex(gains)
            gain = gains[gain_idx]
            xend = gs_results[controller_name]["X"][gain_idx, :, end]
            ω = xend[11:13]
            h = J * ω
            h_end = norm(h)

            max_ylim = max(max_ylim, h_end)
            min_ylim = min(min_ylim, h_end)

            mark = gain_idx == 3 ? "square*" : "*"
            scatteropts = @pgf {mark = mark, only_marks, mark_size = "4pt", mark_options = {color = color_list[controller_idx]}}
            push!(plot_incs, PlotInc(scatteropts, Coordinates([gain], [h_end])))
        end
        x0 = gs_results[controller_name]["X"][1, :, 1]
        ω0 = x0[11:13]
        h0 = norm(J * ω0)

        lineopts = @pgf {no_marks, line_width = "1.25pt", style = "densely dotted", color = "black", opacity = 0.8}
        push!(plot_incs, PlotInc(lineopts, Coordinates(gains, zeros(length(gains)) .+ 0.01 * h0)))


        p = @pgf LogLogAxis(
            {
                set_layers = "standard",
                xmajorgrids,
                ymajorgrids,
                xlabel = L"Gain ($k$)",
                ylabel = L"$\|h\|$ (Nms)",
                legend_pos = "north east",
                title = controller_name,
            },
            plot_incs...,
        )
        push!(plots, p)
        controller_idx += 1
    end
    @pgf groupopts = {
        group_style = {group_size = "3 by 2", horizontal_sep = "0.7in", vertical_sep = "0.9in"},
        height = "2in",
        width = "3.5in",
        ymin = 0.9 * min_ylim,
        ymax = 1.1 * max_ylim
    }
    @pgf gp = GroupPlot(groupopts, plots...)

    pgfsave(joinpath(@__DIR__, "..", "figs", "pdf", "gain_sweep_final_momentum_vs_gain" * file_suffix * ".pdf"), gp)
    pgfsave(joinpath(@__DIR__, "..", "figs", "gain_sweep_final_momentum_vs_gain" * file_suffix * ".tikz"), gp, include_preamble=false)
    return gp
end



datafilename = "gain_sweep_all.jld2"
datapath = joinpath(@__DIR__, "..", "data", datafilename)
data = load(datapath)
gs_results = data["gs_results"]
params = data["params"]

display(pgf_gs_plot_momentum_magnitude_vs_time(gs_results, params; file_suffix=""))
display(pgf_gs_plot_final_momentum_vs_gain(gs_results, params; file_suffix=""))
