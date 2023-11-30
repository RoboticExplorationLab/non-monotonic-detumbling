using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LaTeXStrings
using LinearAlgebra
using JLD2
using Colors
using PGFPlotsX
using StatsBase: Histogram, fit
using Statistics: mean
using Formatting

include("../src/satellite_simulator.jl")

SAVEAS_PDF = true
lineopts = @pgf {no_marks, "very thick", style = "solid"}

color_mode = "_dark_mode"
# color_mode = "" # normal

function color_to_pgf_string(c::Colors.Colorant)
    rgb = convert(Colors.RGB{Float64}, c)
    str = format("rgb,1:red,{:.4f};green,{:.4f};blue,{:.4f}", Colors.red(rgb), Colors.green(rgb), Colors.blue(rgb))
    return raw"{" * str * raw"}"
end

if color_mode == "_dark_mode"
    const color_grid = RGBA(([148, 148, 148, 255] ./ 255)...)
    const color_text = RGBA(([205, 209, 209, 255] ./ 255)...)
    const color_axis = color_text
    const color_bg = RGBA(([0x00, 0x22, 0x39, 0xFF] ./ 255)...)
else
    const color_grid = RGBA(([191, 191, 191, 255] ./ 255)...)
    const color_text = RGBA(([0, 0, 0, 255] ./ 255)...)
    const color_axis = color_text
    const color_bg = RGBA(([255, 255, 255, 255] ./ 255)...)
end

color_grid_pgf = color_to_pgf_string(color_grid)
color_text_pgf = color_to_pgf_string(color_text)
color_axis_pgf = color_to_pgf_string(color_axis)
color_bg_pgf = color_to_pgf_string(color_bg)


function pgf_mc_plot_momentum_magnitude_vs_time(mc_results, params; max_samples=500, file_suffix="")
    Ncontrollers = length(keys(mc_results))
    color_list = distinguishable_colors(Ncontrollers, [RGB(1, 1, 1), RGB(0, 0, 0)], dropseed=true)
    plots = []
    J = params["satellite_model"]["inertia"]
    max_ylim = 0.0
    controller_idx = 1
    for (controller_name, controller_data) in mc_results
        Ntrials = size(mc_results[controller_name]["T"])[1]
        plot_incs = []
        h̄_average = zeros(max_samples + 1)
        t_plot_average = zeros(max_samples + 1)
        for mc_step = 1:Ntrials
            xhist = mc_results[controller_name]["X"][mc_step, :, :]
            uhist = mc_results[controller_name]["U"][mc_step, :, :]
            thist = mc_results[controller_name]["T"][mc_step, 1, :]

            downsample = get_downsample(length(thist), max_samples)
            ω = xhist[11:13, downsample]
            h = J * ω
            h̄ = dropdims(sqrt.(sum(h .* h, dims=1)); dims=1)
            t_plot = thist[downsample] / (60 * 60)
            lineopts_background = @pgf {no_marks, "very thick", style = "solid", color = color_list[controller_idx], opacity = 0.4}
            push!(plot_incs,
                PlotInc(lineopts_background, Coordinates(t_plot, h̄)),
            )
            max_ylim = max(max_ylim, maximum(h̄))

            h̄_average .+= h̄
            t_plot_average .= t_plot
        end
        h̄_average ./= Ntrials
        lineopts_foreground = @pgf {no_marks, line_width = "2pt", style = "dotted", color = color_axis, opacity = 1.0}
        push!(plot_incs,
            PlotInc(lineopts_foreground, Coordinates(t_plot_average, h̄_average)),
        )
        p = @pgf Axis(
            {
                "grid style" = {"color" = color_grid},
                "label style" = {"color" = color_text},
                "title style" = {"color" = color_text},
                "legend style" = {"draw" = color_axis, "fill" = color_bg, "text" = color_text},
                "tick label style" = {"color" = color_text},
                "axis line style" = {"color" = color_axis},
                xmajorgrids,
                ymajorgrids,
                xlabel = "Time (hours)",
                ylabel = L"$\|h\|$ (Nms)",
                legend_pos = "north east",
                title = controller_name,
            },
            plot_incs...,
            Legend([i == length(plot_incs) ? "Average" : "" for i = axes(plot_incs, 1)]))
        push!(plots, p)
        controller_idx += 1
    end
    @pgf groupopts = {
        group_style = {group_size = "3 by 2", horizontal_sep = "0.5in", vertical_sep = "0.9in"},
        height = "2in",
        width = "3.5in",
        ymin = 0,
        ymax = max_ylim
    }
    @pgf gp = GroupPlot(groupopts, plots...)

    if SAVEAS_PDF
        pgfsave(joinpath(@__DIR__, "..", "figs", "pdf", "momentum_magnitude_vs_time" * file_suffix * color_mode * ".pdf"), gp)
    else
        pgfsave(joinpath(@__DIR__, "..", "figs", "momentum_magnitude_vs_time" * file_suffix * color_mode * ".tikz"), gp, include_preamble=false)
    end
end

function pgf_mc_plot_momentum_magnitude_final_histogram(mc_results, params; file_suffix="")
    Ncontrollers = length(keys(mc_results))
    color_list = distinguishable_colors(Ncontrollers, [RGB(1, 1, 1), RGB(0, 0, 0)], dropseed=true)
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
    println("Final momentum bins = $bins")
    plots = []
    max_ylim = 0.0
    for i = 1:size(h_end)[1]
        hist = fit(Histogram, h_end[i, :], bins, closed=:left)
        h_avg_i = mean(h_end[i, :])

        hist_plot = @pgf PlotInc(
            {
                # color = color_list[i],
                draw = color_list[i],
                fill = color_list[i],
                fill_opacity = 1.0
            },
            Table(hist)
        )
        avg_line_plot = [raw"\draw [color=" * color_axis_pgf * "]" * "($h_avg_i, 0) -- ($h_avg_i, $Ntrials);"]

        p = @pgf Axis(
            {
                "grid style" = {"color" = color_grid},
                "label style" = {"color" = color_text},
                "title style" = {"color" = color_text},
                "tick label style" = {"color" = color_text},
                "axis line style" = {"color" = color_axis},
                "ybar interval",
                "xticklabel interval boundaries",
                ymajorgrids,
                xmajorgrids = "false",
                ylabel = "Count",
                xlabel = L"$\|h\|$ (Nms)",
                xtick = range(0, bins[end], 5),
                title = controller_names[i],
                xticklabel = raw"$[\pgfmathprintnumber\tick,\pgfmathprintnumber\nexttick)$",
                xticklabel_style =
                    {
                        font = raw"\footnotesize"
                    },
            },
            hist_plot,
            avg_line_plot,
            [raw"\node ",
                {
                    pin = raw"[draw=" * color_axis_pgf * ",fill=" * color_bg_pgf * ",text=" * color_text_pgf * raw"]right:\footnotesize Average: " * format("{:.2e} Nms", h_avg_i)
                },
                " at ",
                Coordinate(h_avg_i, (Ntrials / 2)),
                "{};"],
        )
        push!(plots, p)
        max_ylim = max(max_ylim, maximum(hist.weights))
    end

    @pgf groupopts = {
        group_style = {group_size = "3 by 2", horizontal_sep = "0.5in", vertical_sep = "0.8in"},
        height = "2in",
        width = "3.5in",
        ymin = 0,
        ymax = max_ylim,
    }
    @pgf gp = GroupPlot(groupopts, plots...)

    if SAVEAS_PDF
        pgfsave(joinpath(@__DIR__, "..", "figs", "pdf", "momentum_magnitude_final_histogram" * file_suffix * color_mode * ".pdf"), gp)
    else
        pgfsave(joinpath(@__DIR__, "..", "figs", "momentum_magnitude_final_histogram" * file_suffix * color_mode * ".tikz"), gp, include_preamble=false)
    end
end

function pgf_mc_plot_detumble_time_cumulative(mc_results, params; terminal_threshold=0.01, file_suffix="")
    Ncontrollers = length(keys(mc_results))
    color_list = distinguishable_colors(Ncontrollers, [RGB(1, 1, 1), RGB(0, 0, 0)], dropseed=true)
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
    println("Detumble time bins = $bins")
    plots = []
    max_ylim = 0.0
    for i = 1:size(t_done)[1]
        t_done_i = sort(t_done[i, :])
        # t_done_i_avg = mean(t_done_i[t_done_i.<t_done_max])
        # t_done_i_avg = mean(t_done_i)
        cum = 100 .* [count(t_done_i .<= bi) for bi in bins] ./ Ntrials
        index_50pct = findfirst(cum .> 50)
        t_done_i_avg = bins[index_50pct]

        cum_plot = @pgf Plot(
            {
                draw = color_list[i],
                fill = color_list[i],
                fill_opacity = 1.0
            },
            # Table(hist)
            Coordinates(bins[1:end-1], cum[1:end-1])
        )
        avg_line_plot = [raw"\draw [color=" * color_axis_pgf * "]" * "($t_done_i_avg, 0) -- ($t_done_i_avg, 100);"]
        avg_line_pin = @pgf [raw"\node ",
            {
                pin = raw"[draw=" * color_axis_pgf * ",fill=" * color_bg_pgf * ",text=" * color_text_pgf * ",align=left]" * (t_done_i_avg > bins[end-1] / 2 ? "left" : "right") * raw":\scriptsize 50\% completed \\ \scriptsize" * format("{:.2f} hours", t_done_i_avg)
            },
            " at ",
            Coordinate(t_done_i_avg, 75),
            "{};"]

        # get index where 100% is hit, end otherwise
        complete_idx_i = findfirst(cum .>= 100)
        complete_idx_i = isnothing(complete_idx_i) || (complete_idx_i > length(cum) - 2) ? length(cum) - 2 : complete_idx_i

        pct_complete = cum[complete_idx_i]
        completed_hours_i = bins[complete_idx_i+1]
        pct_complete_line_plot = [raw"\draw [color=" * color_axis_pgf * "]" * "($completed_hours_i, 0) -- ($completed_hours_i, 100);"]
        pct_complete_yloc = 25 #pct_complete > 50 ? 25 : 75
        pct_complete_label = [raw"\node [draw=" * color_axis_pgf * ",fill=" * color_bg_pgf * ",text=" * color_text_pgf * ",align=left] at " * "($(bins[end-10]), $pct_complete_yloc)" * raw"{\scriptsize " * "$(Int(floor(pct_complete)))" * raw"\%  completed\\ \scriptsize" * format("{:.2f} hours", completed_hours_i) * "};"]

        if pct_complete < 50
            avg_line_pin = []
            avg_line_plot = []
        end

        p = @pgf Axis(
            {
                "grid style" = {"color" = color_grid},
                "label style" = {"color" = color_text},
                "title style" = {"color" = color_text},
                "tick label style" = {"color" = color_text},
                "axis line style" = {"color" = color_axis},
                "ybar interval",
                "xticklabel interval boundaries",
                ymajorgrids,
                xmajorgrids = "false",
                # ylabel = "Percent",
                xlabel = "Time (hours)",
                xtick = range(0, bins[end-1], 5),
                title = controller_names[i],
                xticklabel = raw"$[\pgfmathprintnumber\tick,\pgfmathprintnumber\nexttick)$",
                yticklabel = raw"$\pgfmathprintnumber\tick\,\%$",
                # "xticklabel style" =
                #     {
                #         font = raw"\tiny"
                #     },
            },
            cum_plot,
            avg_line_plot,
            avg_line_pin,
            pct_complete_line_plot,
            pct_complete_label
        )
        push!(plots, p)
    end

    @pgf groupopts = {
        group_style = {group_size = "3 by 2", horizontal_sep = "0.5in", vertical_sep = "0.9in"},
        height = "2in",
        width = "3.5in",
        ymin = 0,
        ymax = 100,
    }
    @pgf gp = GroupPlot(groupopts, plots...)

    if SAVEAS_PDF
        pgfsave(joinpath(@__DIR__, "..", "figs", "pdf", "detumble_time_cumulative" * file_suffix * color_mode * ".pdf"), gp)
    else
        pgfsave(joinpath(@__DIR__, "..", "figs", "detumble_time_cumulative" * file_suffix * color_mode * ".tikz"), gp, include_preamble=false)
    end
end

file_suffix_no_noise = "_no_noise_30deg_s"
datafilename_no_noise = "mc_orbit_varied" * file_suffix_no_noise * ".jld2"
datapath_no_noise = joinpath(@__DIR__, "..", "data", datafilename_no_noise)
data_no_noise = load(datapath_no_noise)
mc_results_no_noise = data_no_noise["mc_results"]
params_no_noise = data_no_noise["params"]

h_thresh = 0.01
pgf_mc_plot_momentum_magnitude_vs_time(mc_results_no_noise, params_no_noise; file_suffix=file_suffix_no_noise, max_samples=200)
pgf_mc_plot_momentum_magnitude_final_histogram(mc_results_no_noise, params_no_noise; file_suffix=file_suffix_no_noise)
pgf_mc_plot_detumble_time_cumulative(mc_results_no_noise, params_no_noise; terminal_threshold=h_thresh, file_suffix=file_suffix_no_noise)


file_suffix_noisy = "_noisy_30deg_s"
datafilename_noisy = "mc_orbit_varied" * file_suffix_noisy * ".jld2"
datapath_noisy = joinpath(@__DIR__, "..", "data", datafilename_noisy)
data_noisy = load(datapath_noisy)
mc_results_noisy = data_noisy["mc_results"]
params_noisy = data_noisy["params"]

pgf_mc_plot_momentum_magnitude_vs_time(mc_results_noisy, params_noisy; file_suffix=file_suffix_noisy, max_samples=200)
pgf_mc_plot_momentum_magnitude_final_histogram(mc_results_noisy, params_noisy; file_suffix=file_suffix_noisy)
pgf_mc_plot_detumble_time_cumulative(mc_results_noisy, params_noisy; terminal_threshold=h_thresh, file_suffix=file_suffix_noisy)