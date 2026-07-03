module Operators

using ..Entities: AqlEntity, Thing, get_property, set_property!,
                  get_attribute, set_attribute!

export roar!, shake!, run!, fall!, become!, emit_scent!, attract_beneficial_insects!

"""
    roar!(source::AqlEntity, target::AqlEntity)

فعل الزئير: يزيد خوف الهدف ويشتت تركيزه بناءً على شدة طاقة وقصور المصدر.
"""
function roar!(source::AqlEntity, target::AqlEntity)
    src_energy, _ = get_property(source, "energy")
    strength = source.mass * (0.5 + 0.5 * src_energy)
    factor = clamp(strength * 0.5, 0.0, 1.0)
    
    fear_amp, fear_phase = get_property(target, "fear")
    awareness_amp, awareness_phase = get_property(target, "awareness")
    
    # محاكاة تأثير التفاعل الفيزيائي برمجياً
    new_fear = (1.0 - factor) * fear_amp + factor * (fear_amp * 6.0)
    new_awareness = (1.0 - factor) * awareness_amp + factor * (awareness_amp * 0.5)
    
    set_property!(target, "fear", new_fear, fear_phase)
    set_property!(target, "awareness", new_awareness, awareness_phase)
end

"""
    shake!(source::AqlEntity, target::AqlEntity)

فعل الاهتزاز: يقلل استقرار الهدف ويزيد حركته.
"""
function shake!(source::AqlEntity, target::AqlEntity)
    src_energy, _ = get_property(source, "energy")
    strength = source.mass * (0.5 + 0.5 * src_energy)
    factor = clamp(strength * 0.5, 0.0, 1.0)
    
    stability_amp, stability_phase = get_property(target, "stability")
    motion_amp, motion_phase = get_property(target, "motion")
    
    new_stability = (1.0 - factor) * stability_amp + factor * (stability_amp * 0.2)
    new_motion = (1.0 - factor) * motion_amp + factor * (motion_amp * 4.0)
    
    set_property!(target, "stability", new_stability, stability_phase)
    set_property!(target, "motion", new_motion, motion_phase)
end

"""
    run!(target::AqlEntity)

فعل الجري/الهرب التلقائي: تبديد الخوف وربطه بحركة نشطة.
"""
function run!(target::AqlEntity)
    set_property!(target, "motion", 0.8, 0.0)
    set_property!(target, "fear", 0.1, 0.0)
end

"""
    fall!(target::AqlEntity)

فعل السقوط والاصطدام: تلاشي الاستقرار، وتفعيل الحركة، وتضرر الهيكل.
"""
function fall!(target::AqlEntity)
    set_property!(target, "stability", 0.0, 0.0)
    set_property!(target, "motion", 0.7, 0.0)
    
    integrity_amp, integrity_phase = get_property(target, "integrity")
    set_property!(target, "integrity", integrity_amp * 0.6, integrity_phase)
end

function become!(source::AqlEntity, target::Thing; property::String="state", value=true, strength::Float64=1.0)
    set_attribute!(target, property, value)
    set_property!(target, "awareness", clamp(strength, 0.0, 1.0), 0.0)
    return target
end

function emit_scent!(source::Thing, target::AqlEntity; strength::Float64=0.7)
    scent = get_attribute(source, "scent", "unknown")
    set_attribute!(source, "last_emitted_scent", scent)
    set_property!(target, "awareness", clamp(strength, 0.0, 1.0), 0.0)
    return target
end

function attract_beneficial_insects!(source::Thing, target::Thing; strength::Float64=0.5)
    set_attribute!(target, "attracted_to", source.name)
    set_attribute!(target, "attraction_kind", "beneficial")
    set_property!(target, "motion", clamp(strength, 0.0, 1.0), 0.0)
    return target
end

end # module Operators
