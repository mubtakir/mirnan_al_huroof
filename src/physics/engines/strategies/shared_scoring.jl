# ═══════════════════════════════════════════════════════
# THE WAVE SCORING FUNCTION
# Every engine contributes: amplitude × exp(i × phase)
# Selection via Born rule: P = |Ψ_total|²
# ═══════════════════════════════════════════════════════
function _score(gen::MirnanGenerator, word::String, used::Set{String},
                all_pv::Vector{<:AbstractVector}, prompt_pv::Vector{<:AbstractVector};
                prev_word=nothing, context_words=nothing)
    wid = get(gen.vocab, word, nothing)
    (wid === nothing || word in used || !_is_generation_candidate(word, context_words)) && return -Inf, WaveContribution()
    w_pv = _pv(gen, word)
    w_norm = norm(w_pv)
    w_norm < 1e-10 && return -Inf, WaveContribution()
    hard_inhibition = _aql_candidate_inhibition(gen, word)
    hard_inhibition >= 0.995 && return -Inf, WaveContribution()

    W = gen.scoring_weights
    waves = WaveContribution[]
    n_ctx = context_words !== nothing ? length(context_words) : 0

    # ═══ K1. KURAMOTO OSCILLATOR ═══
    # Order parameter r measures synchronization of context words
    # Candidate phase aligned with collective → constructive interference
    kuramoto_amp = 0.0
    kuramoto_phase = 0.0
    oscillator_weight = get(W, "oscillator", 0.0)
    if oscillator_weight > 0.0 && n_ctx >= 2 && context_words !== nothing
        ctx_pvs_k = [_pv(gen, cw) for cw in context_words[max(1, end-5):end]]
        Nk = length(ctx_pvs_k)
        re_s = 0.0; im_s = 0.0
        for cpv in ctx_pvs_k
            cn = norm(cpv)
            if cn > 1e-10
                re_s += cpv[1] / cn
                im_s += cpv[2] / cn
            end
        end
        re_s /= Nk; im_s /= Nk
        r = sqrt(re_s^2 + im_s^2)
        cph = atan(im_s, re_s)
        cand_a = w_norm > 1e-10 ? atan(w_pv[2], w_pv[1]) : 0.0
        kuramoto_amp = max(0.0, r * cos(cand_a - cph))
        kuramoto_phase = cph
    end
    push!(waves, WaveContribution(oscillator_weight * kuramoto_amp, kuramoto_phase))

    # ═══ K2. ENTROPY GATE ═══
    # Gates total amplitude based on thermodynamic entropy
    entropy_gate = 1.0
    if n_ctx >= 3 && context_words !== nothing
        ctx_pvs_e = [_pv(gen, cw) for cw in context_words[max(1, end-5):end]]
        try
            S = EntropyGateModule.compute_S(gen.entropy, ctx_pvs_e, w_pv)
            entropy_gate = S < gen.entropy.S_crit ?
                exp(-(gen.entropy.S_crit - S)) :
                1.0 + 0.3 * tanh(S - gen.entropy.S_crit)
        catch e
            @warn "EntropyGate scoring failed: $e"
        end
    end

    # ═══ K3. CAUSAL FLOW ═══
    # Uses K_sem as causal matrix; flow alignment → wave phase
    causal_amp = 0.0
    causal_phase = 0.0
    causal_matrix = gen.K_causal !== nothing ? gen.K_causal : gen.K_sem
    if n_ctx >= 2 && context_words !== nothing && causal_matrix !== nothing
        ctx_ids = [get(gen.vocab, cw, 0) for cw in context_words[max(1, end-5):end]]
        ctx_pvs_cf = [_pv(gen, cw) for cw in context_words[max(1, end-5):end]]
        try
            fr = CausalFlow.compute_flow(gen.causal_flow, w_pv, ctx_pvs_cf, ctx_ids;
                                          causal_matrix=causal_matrix)
            causal_amp = get(fr, "logical_score", 0.0)
            fv = get(fr, "flow_vector", zeros(gen.dim))
            fn = norm(fv)
            fn > 1e-10 && (causal_phase = atan(fv[2], fv[1]))
        catch e
            @warn "CausalFlow scoring failed: $e"
        end
    end
    push!(waves, WaveContribution(get(W, "causal_flow_align", 0.0) * causal_amp, causal_phase))

    # ═══ K4. POTENTIAL CASCADE ═══
    # Gravitational potential wells; 0=attractive, π=repulsive
    cascade_amp = 0.0
    cascade_phase = 0.0
    cascade_weight = get(W, "cascade", 0.0)
    if cascade_weight > 0.0 && n_ctx >= 1 && context_words !== nothing
        ctx_pvs_c = [_pv(gen, cw) for cw in context_words[max(1, end-5):end]]
        ctx_m = [_mass(gen, cw) for cw in context_words[max(1, end-5):end]]
        try
            cs = PotentialCascade.compute_score(gen.cascade, w_pv, ctx_pvs_c, ctx_m,
                                                true; used_words=collect(used),
                                                word_to_pv_fn=w -> _pv(gen, w))
            if cs > 0.0
                cascade_amp = cs
            elseif cs == -Inf
                cascade_amp = 5.0; cascade_phase = π
            end
        catch e
            @warn "PotentialCascade scoring failed: $e"
        end
    end
    push!(waves, WaveContribution(cascade_weight * cascade_amp, cascade_phase))

    # ═══ K5. HETERODYNE ═══
    # Spectral sideband resonance
    het_amp = 0.0
    het_phase = 0.0
    het_weight = get(W, "heterodyne", 0.0)
    if het_weight > 0.0 && n_ctx >= 1 && context_words !== nothing
        ctx_ws = String[string(cw) for cw in context_words[max(1, end-6):end]]
        try
            het_amp = Heterodyne.score_candidate(gen.heterodyne, word, ctx_ws)
            sp = Heterodyne.get_word_spectrum(gen.heterodyne, word)
            length(sp) >= 2 && (het_phase = atan(sp[2], sp[1]))
        catch e
            @warn "Heterodyne scoring failed: $e"
        end
    end
    push!(waves, WaveContribution(het_weight * het_amp, het_phase))

    # ═══ K6. RESONANT CHAIN ═══
    # LC circuit resonance between consecutive words
    rc_amp = 0.0
    rc_phase = 0.0
    rc_weight = get(W, "resonant_chain", 0.0)
    if rc_weight > 0.0 && prev_word !== nothing && context_words !== nothing && length(context_words) >= 1
        try
            mp = _mass(gen, prev_word)
            mc = _mass(gen, word)
            pvp = _pv(gen, prev_word)
            rc_amp = ResonantChain.score_candidate(gen.resonant_chain, mp, mc, pvp, w_pv;
                prev_freqs=isempty(gen.resonant_chain.freq_history) ? nothing :
                gen.resonant_chain.freq_history)
            fr = ResonantChain.pair_freq(gen.resonant_chain, mp, mc, pvp, w_pv)
            rc_phase = 2π * (fr % 1.0)
        catch e
            @warn "ResonantChain scoring failed: $e"
        end
    end
    push!(waves, WaveContribution(rc_weight * rc_amp, rc_phase))

    # ═══ 1. PHASE ALIGNMENT ═══
    target = isempty(all_pv) ? w_pv : all_pv[end]
    tn = isempty(all_pv) ? w_norm : norm(target)
    align_val = w_norm < 1e-10 || tn < 1e-10 ? 0.0 :
                max(0.0, dot(w_pv, target) / (w_norm * tn))
    pa = tn > 1e-10 ? acos(clamp(dot(w_pv, target) / (w_norm * tn), -1.0, 1.0)) : 0.0
    push!(waves, WaveContribution(get(W, "align", 0.0) * align_val, pa))

    # ═══ 2. PROMPT ALIGNMENT ═══
    pav = 0.0; pp = 0.0
    if !isempty(prompt_pv)
        for p in prompt_pv
            pn = norm(p)
            if w_norm > 1e-10 && pn > 1e-10
                s = max(0.0, dot(w_pv, p) / (w_norm * pn))
                pav += s; pp += acos(clamp(s, -1.0, 1.0))
            end
        end
        pav /= length(prompt_pv); pp /= length(prompt_pv)
    end
    push!(waves, WaveContribution(get(W, "prompt_align", 0.0) * pav, pp))

    # ═══ 3. DIVERSITY — angular distance from last word ═══
    dv = 1.0
    if !isempty(used)
        pu = collect(used)[max(1, end-6):end]
        dv = 1.0 - count(x -> x == word, pu) / length(pu)
    end
    dp = if n_ctx >= 1 && context_words !== nothing
        lp = _pv(gen, context_words[end])
        ln = norm(lp)
        ln > 1e-10 ? acos(clamp(dot(w_pv, lp) / (w_norm * ln), -1.0, 1.0)) : 0.0
    else
        0.0
    end
    push!(waves, WaveContribution(get(W, "diversity", 0.0) * dv, dp))

    # ═══ 4. GRAVITY ═══
    gv = 0.0; gp = 0.0
    if context_words !== nothing && !isempty(context_words)
        mw = _mass(gen, word)
        for cw in context_words[max(1, end-5):end]
            cp = _pv(gen, cw); mc = _mass(gen, cw)
            dsq = max(0.0, w_norm^2 + norm(cp)^2 - 2.0 * dot(w_pv, cp))
            r = sqrt(dsq) + 1e-6
            gv += GRAVITY_G * mw * mc / (r^2 + 0.01)
            gp += compute_phase_from_gravity(w_pv, cp, mw, mc)
        end
        gp /= min(5, length(context_words))
    end
    push!(waves, WaveContribution(get(W, "gravity", 0.0) * gv, gp))

    # ═══ 5. SYNTAX ═══
    sv = 0.0; sp = 0.0
    if prev_word !== nothing
        ps = _syn(gen, prev_word); cs = _syn(gen, word)
        sn = norm(ps) * norm(cs)
        if sn > 1e-10
            sv = max(0.0, dot(cs, ps) / sn)
            sp = compute_phase_from_syntax(cs)
        end
    end
    push!(waves, WaveContribution(get(W, "syntax", 0.0) * sv, sp))

    # ═══ 6. DENSITY RESONANCE ═══
    density_weight = get(W, "density_resonance", 0.0)
    dr = 0.0
    dp6 = 0.0
    if density_weight > 0.0
        dr = DensityMatrix.resonance(gen.density_matrix, w_pv)
        dp6 = gen.density_matrix.rho !== nothing ?
            compute_phase_from_density_matrix(gen.density_matrix.rho, w_pv) : 0.0
    end
    push!(waves, WaveContribution(density_weight * dr, dp6))

    # ═══ 7. DCCF ═══
    dccf_weight = get(W, "dccf", 0.0)
    dccf_v = if dccf_weight > 0.0 && context_words !== nothing && !isempty(context_words)
        ctx7 = [_pv(gen, cw) for cw in context_words[max(1, end-5):end]]
        get_context_boost(gen.dccf, w_pv, ctx7)
    else; 0.0; end
    dccf_p = if dccf_weight > 0.0 && context_words !== nothing && !isempty(context_words)
        compute_phase_from_ppm(w_pv, _pv(gen, context_words[end]))
    else; 0.0; end
    push!(waves, WaveContribution(dccf_weight * dccf_v, dccf_p))

    # ═══ 8. PPM ═══
    pm_v = score(gen.prompt_field, w_pv)
    pm_p = gen.prompt_field.active && norm(gen.prompt_field.field) > 1e-10 ?
        compute_phase_from_ppm(gen.prompt_field.field, w_pv) : 0.0
    prompt_len = length(prompt_pv)
    ppm_factor = prompt_len <= 3 ? 0.90 : prompt_len <= 8 ? 0.70 : 0.45
    push!(waves, WaveContribution(get(W, "ppm", 0.0) * ppm_factor * pm_v, pm_p))

    # ═══ 9. ROOT AFFINITY — phase from root overlap ═══
    ra = _root_affinity(word, context_words)
    rp = if n_ctx >= 1 && context_words !== nothing
        wr = Set(_safe_root(word)); ba = 0.0
        for cw in context_words[max(1, end-3):end]
            ba = max(ba, _set_overlap(wr, Set(_safe_root(cw))))
        end
        ba * π
    else; 0.0; end
    push!(waves, WaveContribution(get(W, "root_affinity", 0.0) * ra, rp))

    # ═══ 10. SURFACE AFFINITY — phase from surface overlap ═══
    surface_weight = get(W, "surface_affinity", 0.0)
    sa = surface_weight > 0.0 ? _surface_affinity(word, context_words) : 0.0
    sap = if surface_weight > 0.0 && n_ctx >= 1 && context_words !== nothing
        ws = _surface_chars(word); bs = 0.0
        for cw in context_words[max(1, end-3):end]
            bs = max(bs, _set_overlap(ws, _surface_chars(cw)))
        end
        bs * π
    else; 0.0; end
    push!(waves, WaveContribution(surface_weight * sa, sap))

    # 11. CONTEXT TENSION: local phase lock plus lexical anchor.
    ct_amp, ct_phase = _context_tension(gen, word, context_words, w_pv)
    push!(waves, WaveContribution(get(W, "context_tension", 0.0) * ct_amp, ct_phase))

    # 12. AL-AQL GUIDANCE: ideas, relations, events, goals, and exceptions
    # contribute a directed semantic field before free statistical drift.
    aql_amp = _aql_candidate_bias(gen, word)
    aql_phase = aql_amp > 0.0 ? 0.25π * (1.0 - aql_amp) : 0.0
    push!(waves, WaveContribution(get(W, "aql_guidance", 0.0) * aql_amp, aql_phase))

    # 13. AL-AQL INHIBITION: curated negative rules, rejected rules, and
    # rejection lessons create an explicit destructive field.
    inh_amp = hard_inhibition
    push!(waves, WaveContribution(get(W, "aql_inhibition", 0.0) * inh_amp, pi))

    # ═══ 11.b POLARITY HARMONY ═══
    pol_weight = get(W, "polarity", 0.0)
    if pol_weight > 0.0 && n_ctx >= 1 && context_words !== nothing
        cw_pol = PolarityField.word_polarity_vector(word)
        pol_sum = 0.0; pol_n = 0
        for cw in context_words[max(1, end-3):end]
            cw_p = PolarityField.word_polarity_vector(cw)
            n1 = sqrt(sum(cw_p[i]^2 for i in 1:3))
            n2 = sqrt(sum(cw_pol[i]^2 for i in 1:3))
            if n1 > 1e-10 && n2 > 1e-10
                p_sim = sum(cw_pol[i]*cw_p[i] for i in 1:3) / (n1 * n2)
                pol_sum += max(0.0, p_sim)
                pol_n += 1
            end
        end
        if pol_n > 0
            pol_amp = pol_sum / pol_n
            pol_phase = 0.25π * (1.0 - pol_amp)
            push!(waves, WaveContribution(pol_weight * pol_amp, pol_phase))
        end
    end

    # ═══ 11.c ANATOMICAL SIMILARITY ═══
    anat_weight = get(W, "anatomical", 0.0)
    if anat_weight > 0.0 && n_ctx >= 1 && context_words !== nothing
        cw_anat = AnatomicalField.word_anatomical_vector(word)
        a_sum = 0.0; a_n = 0
        for cw in context_words[max(1, end-3):end]
            ca = AnatomicalField.word_anatomical_vector(cw)
            n1 = sqrt(sum(ca[i]^2 for i in 1:7))
            n2 = sqrt(sum(cw_anat[i]^2 for i in 1:7))
            if n1 > 1e-10 && n2 > 1e-10
                a_sim = sum(cw_anat[i]*ca[i] for i in 1:7) / (n1 * n2)
                a_sum += max(0.0, a_sim)
                a_n += 1
            end
        end
        if a_n > 0
            a_amp = a_sum / a_n
            a_phase = 0.25π * (1.0 - a_amp)
            push!(waves, WaveContribution(anat_weight * a_amp, a_phase))
        end
    end

    # ═══ 11.d OBJECTIVITY ═══
    obj_weight = get(W, "objectivity", 0.0)
    if obj_weight > 0.0 && n_ctx >= 1 && context_words !== nothing
        cw_obj = ObjectivityField.word_objectivity(word)
        o_sum = 0.0; o_n = 0
        for cw in context_words[max(1, end-3):end]
            co = ObjectivityField.word_objectivity(cw)
            o_sim = 1.0 - abs(cw_obj - co)
            o_sum += max(0.0, o_sim)
            o_n += 1
        end
        if o_n > 0
            o_amp = o_sum / o_n
            o_phase = π * (1.0 - cw_obj)  # low objectivity → opposite phase (subjective)
            push!(waves, WaveContribution(obj_weight * o_amp, o_phase))
        end
    end

    # ═══ 12. K_SEM ═══
    ksv = 0.0; ksp = 0.0
    ks = get(gen.k_sem_config, "strength", 1.0)
    kt = max(get(gen.k_sem_config, "temperature", 1.0), 0.01)
    if gen.K_sem !== nothing
        mkv = 0.0; bk = 0.0
        for cw in (context_words !== nothing ? context_words[max(1, end-3):end] : String[])
            cid = get(gen.vocab, cw, 0)
            cid > 0 && cid <= size(gen.K_sem, 1) || continue
            wid2 = get(gen.vocab, word, 0)
            wid2 > 0 || continue
            kv = gen.K_sem[wid2, cid]
            ka = ks * abs(kv) ^ (1.0 / kt)
            if ka > mkv; mkv = ka; bk = kv; end
        end
        ksv = mkv; ksp = compute_phase_from_ksem(bk)
    end
    push!(waves, WaveContribution(get(W, "k_sem", 0.0) * ksv, ksp))

    # ═══ 12.b RETRIEVAL-AUGMENTED PHYSICAL GENERATION (RAPG) ═══
    ret_amp = 0.0
    ret_phase = 0.0
    ret_weight = get(W, "kb_knowledge", 0.0)
    n_ret = isdefined(gen, :retrieved_pvs) ? length(gen.retrieved_pvs) : 0
    if ret_weight > 0.0 && n_ret > 0
        re_s = 0.0; im_s = 0.0
        for i in 1:n_ret
            ret_pv = gen.retrieved_pvs[i]
            query_sim = gen.retrieved_similarities[i]
            
            word_sim = 0.0
            rn = norm(ret_pv)
            if w_norm > 1e-10 && rn > 1e-10
                word_sim = dot(w_pv, ret_pv) / (w_norm * rn)
            end
            
            amp_contrib = query_sim * max(0.0, word_sim)
            phase_contrib = acos(clamp(word_sim, -1.0, 1.0))
            
            re_s += amp_contrib * cos(phase_contrib)
            im_s += amp_contrib * sin(phase_contrib)
        end
        re_s /= n_ret
        im_s /= n_ret
        ret_amp = sqrt(re_s^2 + im_s^2)
        ret_phase = atan(im_s, re_s)
    end
    push!(waves, WaveContribution(ret_weight * ret_amp, ret_phase))


    # ═══ 12. TRAJECTORY ALIGNMENT ═══
    trj_amp = 0.0; trj_phase = 0.0
    if !isempty(gen.trajectory.milestones)
        try
            trj_score = trajectory_score(gen.trajectory, w_pv, n_ctx + 1)
            if trj_score > 0.0
                trj_amp = trj_score
                trj_phase = acos(clamp(trj_score, -1.0, 1.0))
            end
        catch e
            @warn "Trajectory alignment failed: $e"
        end
    end
    push!(waves, WaveContribution(get(W, "trajectory", 0.0) * trj_amp, trj_phase))

    # ═══ 13. ARCHITECT (open/body/close awareness) ═══
    arc_amp = 0.0; arc_phase = 0.0
    try
        arc_score = architect_score(gen.architect, n_ctx + 1, max(n_ctx + 1, 10), word)
        if arc_score > 0.0
            arc_amp = arc_score
            arc_phase = arc_score * π
        end
    catch e
        @warn "Architect scoring failed: $e"
    end
    push!(waves, WaveContribution(get(W, "architect", 0.0) * arc_amp, arc_phase))

    # ═══ SUPERPOSITION + ENTROPY GATE + BORN RULE ═══
    total = wave_superposition(waves)
    total = WaveContribution(total.amplitude * entropy_gate, total.phase)
    probability = born_rule(total)

    return probability, total, entropy_gate
end

function _shape_syntax_bonus(gen::MirnanGenerator, word::String, expected_role::AbstractString)
    syn = _syn(gen, word)
    actual_phase = compute_phase_from_syntax(syn)
    expected_phase = token_role_phase(expected_role)
    bonus = cos(actual_phase - expected_phase)
    isapprox(expected_phase, 0.0, atol=0.001) && (bonus = max(0.0, bonus))
    return clamp(0.12 * bonus, 0.0, 0.35)
end

function _semantic_candidate_bonus(w::String, semantic_field)
    semantic_field isa Dict || return 0.0
    get(semantic_field, "active", false) || return 0.0
    confidence = get(semantic_field, "confidence", 0.0)
    confidence < 0.25 && return 0.0
    terms = get(semantic_field, "target_terms", String[])
    w in terms && return clamp(0.35 * confidence, 0.0, 0.35)
    movement = get(semantic_field, "movement", "")
    movement in ("definition", "method", "cause_effect", "comparison") && return 0.05 * confidence
    return 0.0
end
