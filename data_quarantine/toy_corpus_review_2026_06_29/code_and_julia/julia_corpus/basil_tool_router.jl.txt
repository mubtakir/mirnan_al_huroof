module ToolRouter

export ToolMeta, Router, register!, execute, get_tools_listing

struct ToolMeta
    name::String
    func::Function
    category::String
    is_safe::Bool
end

mutable struct Router
    workspace_root::String
    tools::Dict{String, ToolMeta}
    agent_instance::Any
end

function Router(workspace_root::String, agent_instance::Any=nothing)
    router = Router(workspace_root, Dict{String, ToolMeta}(), agent_instance)
    return router
end

function register!(router::Router, name::String, func::Function, category::String="general"; is_safe::Bool=true)
    if haskey(router.tools, name)
        @warn "Tool '$name' is already registered. Overwriting..."
    end
    router.tools[name] = ToolMeta(name, func, category, is_safe)
end

function execute(router::Router, tool_name::String, args::Vector{Any})::String
    if !haskey(router.tools, tool_name)
        return "[ERROR] 🛑 أداة غير مسجلة: $tool_name"
    end
    
    meta = router.tools[tool_name]
    try
        result = meta.func(args...)
        return string(result)
    catch e
        @error "Tool crash on execution: $tool_name" exception=(e, catch_backtrace())
        return "[TOOL_CRASH] 🛑 $tool_name failed: $e"
    end
end

function get_tools_listing(router::Router)::String
    categorized = Dict{String, Vector{ToolMeta}}()
    for t in values(router.tools)
        if !haskey(categorized, t.category)
            categorized[t.category] = ToolMeta[]
        end
        push!(categorized[t.category], t)
    end
    
    listing = "### 🛠️ Sovereign Engineering Toolbox:\n"
    for (cat, tools) in categorized
        listing *= "\n**Category: $(uppercasefirst(cat))**\n"
        for t in tools
            listing *= "- `$(t.name)`: أداة من فئة $(t.category)\n"
        end
    end
    return listing
end

end # module ToolRouter
