module TerminalExecutor

export TerminalExecutor, execute_command

struct TerminalExecutor
    workspace_root::String
end

_has_arabic(text::AbstractString) = occursin(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", text)

function _fix_arabic_output(text::String, workspace_root::String)
    isempty(text) && return text
    get(ENV, "MIRNAN_FIX_ARABIC_TERMINAL", "1") in ("0", "false", "False") && return text
    _has_arabic(text) || return text

    root = abspath(workspace_root)
    script_candidates = [
        joinpath(root, "models", "mirnan", "scripts", "fix-arabic.jl"),
        joinpath(root, "scripts", "fix-arabic.jl"),
        joinpath(dirname(root), "models", "mirnan", "scripts", "fix-arabic.jl"),
    ]
    script_index = findfirst(isfile, script_candidates)
    script_index === nothing && return text
    script = script_candidates[script_index]

    tmp = tempname()
    try
        write(tmp, text)
        args = String[]
        if get(ENV, "MIRNAN_FIX_ARABIC_FORCE", "1") in ("1", "true", "True")
            push!(args, "--force")
        end
        return read(`$(Base.julia_cmd()) $script $(args...) $tmp`, String)
    catch
        return text
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
end

function execute_command(term::TerminalExecutor, cmd_str::String)::String
    shell = Sys.iswindows() ? ["powershell", "-ExecutionPolicy", "Bypass", "-Command"] : ["bash", "-c"]
    full_cmd = Cmd([shell..., cmd_str], dir=term.workspace_root)
    
    out_buf = IOBuffer()
    err_buf = IOBuffer()
    
    try
        proc = run(pipeline(full_cmd, stdout=out_buf, stderr=err_buf), wait=true)
        output = _fix_arabic_output(String(take!(out_buf)), term.workspace_root)
        errors = _fix_arabic_output(String(take!(err_buf)), term.workspace_root)
        
        result = output
        if !isempty(errors)
            result *= "\n[STDERR]\n" * errors
        end
        return isempty(result) ? "[SUCCESS] Command completed with no output." : result
    catch e
        output = _fix_arabic_output(String(take!(out_buf)), term.workspace_root)
        errors = _fix_arabic_output(String(take!(err_buf)), term.workspace_root)
        result = output
        if !isempty(errors)
            result *= "\n[STDERR]\n" * errors
        end
        return "[ERROR] Command execution failed: $e\nOutput:\n" * result
    end
end

end # module TerminalExecutor
