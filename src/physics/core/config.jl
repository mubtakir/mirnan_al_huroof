"""
MirnanConfig — مركز تحميل وتخزين إعدادات config.yaml
يقرا الملف مرة واحدة ويوفر القيم لكل المحركات.
"""
module MirnanConfig

import YAML

export MirnanCfg, load_config, get_section

struct MirnanCfg
    data::Dict{String,Any}
end

function load_config(path::String="")
    if isempty(path)
        path = joinpath(@__DIR__, "..", "..", "..", "config.yaml")
    end
    data = Dict{String,Any}()
    try
        if isfile(path)
            data = YAML.load_file(path)
        else
            @warn "config.yaml not found: $path — using defaults"
        end
    catch e
        @warn "Failed to load config.yaml: $e — using defaults"
    end
    return MirnanCfg(data)
end

get_section(cfg::MirnanCfg, section::String) = get(cfg.data, section, Dict{String,Any}())
get_val(cfg::MirnanCfg, section::String, key::String, default) = get(get_section(cfg, section), key, default)
Base.getindex(cfg::MirnanCfg, section::String) = get_section(cfg, section)

function Base.show(io::IO, cfg::MirnanCfg)
    sections = sort(collect(keys(cfg.data)))
    print(io, "MirnanCfg($(length(sections)) sections: $(join(sections[1:min(5,end)], ", "))...)")
end

end # module MirnanConfig
