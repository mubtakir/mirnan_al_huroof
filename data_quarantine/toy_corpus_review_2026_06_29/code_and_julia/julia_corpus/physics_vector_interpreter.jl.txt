"""VectorInterpreter — مفسر المتجهات (تفسير الأبعاد المهيمنة)."""
module VectorInterpreter
using LinearAlgebra
export interpret_vector, DIM_NAMES

const DIM_NAMES = ["concentration","internal_external","stability_motion","density","temperature","time_accumulation","time_peak","time_discharge","motion_linear","motion_rotary","motion_pulse","motion_stretch","motion_slip","motion_air","axis_v","mass","hardness_solid","penetration","charge","reference_self","space_extensionality","time_causality"]

function interpret_vector(pv::AbstractVector; top_k=5)
    abs_pv = abs.(Float64.(pv[1:min(22,end)]))
    top_idx = sortperm(abs_pv; rev=true)[1:min(top_k, end)]
    return [(DIM_NAMES[i], Float64(pv[i])) for i in top_idx]
end
end
