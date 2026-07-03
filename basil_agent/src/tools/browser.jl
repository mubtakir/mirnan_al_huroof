module BrowserAgent

using JSON3

export browser_navigate, browser_click, browser_fill, browser_get_text, browser_get_links, browser_screenshot

const HELPER_PATH = joinpath(@__DIR__, "browser_helper.py")

function _run_helper(workspace_root::String, cmd_args::Vector{String})::String
    cmd = Cmd(["python", HELPER_PATH, workspace_root, cmd_args...])
    out_buf = IOBuffer()
    err_buf = IOBuffer()
    
    try
        run(pipeline(cmd, stdout=out_buf, stderr=err_buf))
        out_str = String(take!(out_buf))
        err_str = String(take!(err_buf))
        
        # Parse output as JSON
        res = JSON3.read(out_str, Dict)
        if get(res, "status", "error") == "success"
            if haskey(res, "content")
                return "[BROWSER SUCCESS]\n" * res["content"]
            elseif haskey(res, "links")
                links_str = join(["- [$(l["text"])]($(l["href"]))" for l in res["links"]], "\n")
                return "[BROWSER SUCCESS] Page Links:\n" * links_str
            elseif haskey(res, "path")
                return "[BROWSER SUCCESS] Screenshot saved to: $(res["path"])"
            else
                return "[BROWSER SUCCESS] " * get(res, "message", "Action completed successfully.")
            end
        else
            return "[BROWSER ERROR] " * get(res, "message", "Unknown browser error.")
        end
    catch e
        out_str = String(take!(out_buf))
        err_str = String(take!(err_buf))
        return "[BROWSER CRASH] Failed to execute helper: $e\nStdout: $out_str\nStderr: $err_str"
    end
end

function browser_navigate(workspace_root::String, url::String)::String
    return _run_helper(workspace_root, ["navigate", url])
end

function browser_click(workspace_root::String, selector::String)::String
    return _run_helper(workspace_root, ["click", selector])
end

function browser_fill(workspace_root::String, selector::String, value::String)::String
    return _run_helper(workspace_root, ["fill", selector, value])
end

function browser_get_text(workspace_root::String)::String
    return _run_helper(workspace_root, ["text"])
end

function browser_get_links(workspace_root::String)::String
    return _run_helper(workspace_root, ["links"])
end

function browser_screenshot(workspace_root::String, label::String="state")::String
    return _run_helper(workspace_root, ["screenshot", label])
end

end # module BrowserAgent
