# using Pkg
# pkg"registry add General https://github.com/legend-exp/LegendJuliaRegistry.git"
# Pkg.add(["LegendGeSim", "Plots"])

using LegendGeSim
using Plots

detector_metadata_filename = "V08682A.json"

# --------------------------------------------------------------------------------------
# --------------------------------------- PART 1 ---------------------------------------
detector = LegendGeSim.LEGEND_SolidStateDetector(detector_metadata_filename)
p = plot(detector, aspect_ratio=1.0, camera=(20,20), size=(600,600), title="V08682A", 
    left_margin = -10Plots.mm, right_margin = -10Plots.mm,
    top_margin = -8Plots.mm, bottom_margin = -8Plots.mm)

for series in p.series_list
    lbl = series[:label]
    if lbl isa AbstractString && occursin("(id:", lbl)
        series[:label] = replace(lbl, r"\s*\(id:\s*\d+\)" => "")
    end
end
savefig(p, "geometry_V08682A.svg")
# --------------------------------------------------------------------------------------


# plot(detector, aspect_ratio=1.0, camera=(20,20), size=(450,450), title="V08682A", 
#     left_margin = -10Plots.mm, right_margin = -10Plots.mm,
#     top_margin = -8Plots.mm, bottom_margin = -8Plots.mm)

environment_settings = Dict(
    "crystal_temperature_in_K" => 77,   # temperatura operativa standard per HPGe
    "medium" => "vacuum"
)

simulation_settings = Dict(
    "method" => "SSD",
    "cached_name" => "V08682A_cache"   # stringa non vuota => cache dei risultati su disco
)

sim = LegendGeSim.simulate_fields(
    "V08682A.json",
    Dict("crystal_temperature_in_K" => 77, "medium" => "vacuum"),
    Dict("method" => "SSD", "cached_name" => "V08682A_cache");
    overwrite = false   # riusa cache\V08682A_V08682A_cache_ssd_fields.h5f invece di ricalcolare
)

mycmap = cgrad(["#7F1C6E", "#013162", "#006AA4", "#FF9100", "#FE2F33"])

for c in sim.detector.contacts
    println("id=", c.id, "  name=", c.name, "  potential=", c.potential)
end

# Weighting potential del point contact (elettrodo di lettura)
p_wp_pc = plot(sim.weighting_potentials[1], full_det = true,
                title = "Weighting potential — point contact", c= :plasma)
savefig(p_wp_pc, "weighting_potential_pointcontact.pdf")

# Weighting potential del mantello/massa
p_wp_mantle = plot(sim.weighting_potentials[2], full_det = true,
                     title = "Weighting potential — mantle", c= :plasma)
savefig(p_wp_mantle, "weighting_potential_mantle.pdf")

# Entrambi affiancati, comodo per la tesi
p_both = plot(
    plot(sim.weighting_potentials[1], full_det = true, title = "Point contact", c=:plasma),
    plot(sim.weighting_potentials[2], full_det = true, title = "Mantle", c=:plasma),
    layout = (1, 2), size = (900, 450)
)
savefig(p_both, "weighting_potentials_both.pdf")