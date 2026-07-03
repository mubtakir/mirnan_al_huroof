module Taxonomy

using JSON
using ..Entities: AqlEntity, Thing, PROPERTY_MAP,
                  get_property, get_attribute, set_attribute!, set_property!

export ConceptClass, ConceptTaxonomy, default_aql_taxonomy,
       register_class!, add_subclass!, assign_class!, assigned_classes,
       class_lineage, inherited_attributes, apply_taxonomy!, is_a,
       explain_classification, known_classes, load_taxonomy, save_taxonomy,
       DEFAULT_TAXONOMY_FILE

const DEFAULT_TAXONOMY_FILE = joinpath(@__DIR__, "..", "..", "..", "data", "al_aql_taxonomy.json")

struct ConceptClass
    name::String
    parents::Vector{String}
    attributes::Dict{String,Any}
    property_defaults::Dict{String,Float64}
    members::Vector{String}
    abstract_key::Bool
end

mutable struct ConceptTaxonomy
    classes::Dict{String,ConceptClass}
    entity_classes::Dict{String,Vector{String}}
end

ConceptTaxonomy() = ConceptTaxonomy(Dict{String,ConceptClass}(), Dict{String,Vector{String}}())

function _as_strings(values)
    result = String[]
    values === nothing && return result
    if values isa AbstractString
        push!(result, String(values))
    elseif values isa AbstractVector
        for value in values
            value === nothing && continue
            push!(result, String(value))
        end
    else
        push!(result, String(values))
    end
    return result
end

function _push_unique!(items::Vector{String}, value::AbstractString)
    text = String(value)
    isempty(text) && return items
    text in items || push!(items, text)
    return items
end

function _merge_unique!(items::Vector{String}, values)
    for value in _as_strings(values)
        _push_unique!(items, value)
    end
    return items
end

function register_class!(tax::ConceptTaxonomy, name::AbstractString;
                         parent::Union{Nothing,AbstractString}=nothing,
                         parents=nothing,
                         attributes::Dict{String,<:Any}=Dict{String,Any}(),
                         properties::Dict{String,<:Real}=Dict{String,Float64}(),
                         members=nothing,
                         abstract_key::Bool=true)
    class_name = String(name)
    parent_names = String[]
    parent !== nothing && push!(parent_names, String(parent))
    _merge_unique!(parent_names, parents)

    attrs = Dict{String,Any}(attributes)
    props = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in properties)
    member_names = _as_strings(members)

    tax.classes[class_name] = ConceptClass(class_name, parent_names, attrs, props,
                                           member_names, abstract_key)
    return tax.classes[class_name]
end

function add_subclass!(tax::ConceptTaxonomy, child::AbstractString, parent::AbstractString;
                       attributes::Dict{String,<:Any}=Dict{String,Any}(),
                       properties::Dict{String,<:Real}=Dict{String,Float64}(),
                       members=nothing,
                       abstract_key::Bool=true)
    return register_class!(tax, child; parent=parent, attributes=attributes,
                           properties=properties, members=members,
                           abstract_key=abstract_key)
end

known_classes(tax::ConceptTaxonomy) = sort(collect(keys(tax.classes)))

function _class_memberships(tax::ConceptTaxonomy, entity_name::AbstractString)
    name = String(entity_name)
    result = String[]
    for (class_name, cls) in tax.classes
        name in cls.members && push!(result, class_name)
    end
    return result
end

function assigned_classes(tax::ConceptTaxonomy, entity_name::AbstractString)
    name = String(entity_name)
    classes = copy(get(tax.entity_classes, name, String[]))
    _merge_unique!(classes, _class_memberships(tax, name))
    return classes
end

function assigned_classes(tax::ConceptTaxonomy, e::AqlEntity)
    classes = assigned_classes(tax, e.name)
    if e isa Thing
        stored = get_attribute(e, "__classes", String[])
        _merge_unique!(classes, stored)
        haskey(tax.classes, e.kind) && _push_unique!(classes, e.kind)
    end
    return classes
end

function assign_class!(tax::ConceptTaxonomy, entity_name::AbstractString, class_name::AbstractString)
    cname = String(class_name)
    haskey(tax.classes, cname) || error("Unknown al_aql class: $cname")
    current = get!(tax.entity_classes, String(entity_name), String[])
    _push_unique!(current, cname)
    return current
end

function assign_class!(tax::ConceptTaxonomy, e::AqlEntity, class_name::AbstractString; apply::Bool=true)
    current = assign_class!(tax, e.name, class_name)
    if e isa Thing
        stored = _as_strings(get_attribute(e, "__classes", String[]))
        _push_unique!(stored, class_name)
        set_attribute!(e, "__classes", stored)
    end
    apply && apply_taxonomy!(tax, e)
    return current
end

function class_lineage(tax::ConceptTaxonomy, class_name::AbstractString; include_self::Bool=true)
    cname = String(class_name)
    visited = Set{String}()
    result = String[]

    function visit(name::String)
        (name in visited || !haskey(tax.classes, name)) && return
        push!(visited, name)
        for parent in tax.classes[name].parents
            visit(parent)
        end
        push!(result, name)
    end

    visit(cname)
    include_self || filter!(x -> x != cname, result)
    return result
end

function inherited_attributes(tax::ConceptTaxonomy, class_name::AbstractString)
    attrs = Dict{String,Any}()
    for cname in class_lineage(tax, class_name)
        merge!(attrs, tax.classes[cname].attributes)
    end
    return attrs
end

function _inherited_properties(tax::ConceptTaxonomy, class_name::AbstractString)
    props = Dict{String,Float64}()
    for cname in class_lineage(tax, class_name)
        merge!(props, tax.classes[cname].property_defaults)
    end
    return props
end

function is_a(tax::ConceptTaxonomy, class_name::AbstractString, expected_class::AbstractString)
    expected = String(expected_class)
    return expected in class_lineage(tax, class_name)
end

function is_a(tax::ConceptTaxonomy, e::AqlEntity, expected_class::AbstractString)
    for class_name in assigned_classes(tax, e)
        is_a(tax, class_name, expected_class) && return true
    end
    return false
end

function apply_taxonomy!(tax::ConceptTaxonomy, e::AqlEntity)
    classes = assigned_classes(tax, e)
    if e isa Thing
        set_attribute!(e, "__classes", classes)
    end

    for class_name in classes
        haskey(tax.classes, class_name) || continue
        for (key, value) in inherited_attributes(tax, class_name)
            if e isa Thing && !haskey(e.attributes, key)
                set_attribute!(e, key, value)
            end
        end
        if e isa Thing
            for (key, value) in _inherited_properties(tax, class_name)
                if key in keys(PROPERTY_MAP)
                    current, phase = get_property(e, key)
                    isapprox(current, 0.1; atol=1e-12) && set_property!(e, key, value, phase)
                end
            end
        end
    end
    return e
end

function explain_classification(tax::ConceptTaxonomy, e::AqlEntity)
    parts = String[]
    for class_name in assigned_classes(tax, e)
        lineage = class_lineage(tax, class_name)
        isempty(lineage) && continue
        push!(parts, join(lineage, " > "))
    end
    return parts
end

function _dict_string_any(raw)
    raw === nothing && return Dict{String,Any}()
    result = Dict{String,Any}()
    for (key, value) in raw
        result[String(key)] = value
    end
    return result
end

function _dict_string_float(raw)
    raw === nothing && return Dict{String,Float64}()
    result = Dict{String,Float64}()
    for (key, value) in raw
        value isa Number || continue
        result[String(key)] = Float64(value)
    end
    return result
end

function load_taxonomy(path::AbstractString=DEFAULT_TAXONOMY_FILE)
    tax = ConceptTaxonomy()
    isfile(path) || return tax

    data = JSON.parsefile(String(path))
    classes = get(data, "classes", Any[])
    for item in classes
        name = get(item, "name", nothing)
        name === nothing && continue
        register_class!(tax, String(name);
            parents=_as_strings(get(item, "parents", String[])),
            attributes=_dict_string_any(get(item, "attributes", Dict{String,Any}())),
            properties=_dict_string_float(get(item, "properties", Dict{String,Any}())),
            members=_as_strings(get(item, "members", String[])),
            abstract_key=Bool(get(item, "abstract_key", true)))
    end

    return tax
end

function _taxonomy_data(tax::ConceptTaxonomy)
    classes = Any[]
    for class_name in known_classes(tax)
        cls = tax.classes[class_name]
        push!(classes, Dict{String,Any}(
            "name" => cls.name,
            "parents" => cls.parents,
            "attributes" => cls.attributes,
            "properties" => cls.property_defaults,
            "members" => cls.members,
            "abstract_key" => cls.abstract_key,
        ))
    end
    return Dict{String,Any}(
        "version" => 1,
        "description" => "al_aql abstract taxonomy database",
        "classes" => classes,
    )
end

function save_taxonomy(tax::ConceptTaxonomy, path::AbstractString=DEFAULT_TAXONOMY_FILE)
    mkpath(dirname(String(path)))
    open(String(path), "w") do io
        JSON.print(io, _taxonomy_data(tax), 2)
    end
    return String(path)
end

default_aql_taxonomy() = load_taxonomy(DEFAULT_TAXONOMY_FILE)

end # module Taxonomy
