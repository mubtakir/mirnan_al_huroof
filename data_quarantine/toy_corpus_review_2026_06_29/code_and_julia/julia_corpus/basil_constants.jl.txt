module Constants

using YAML

export WORKSPACE_ROOT, AGENT_NAME, AGENT_VERSION, LLM_PROVIDER, OPENROUTER_API_KEY,
       REMOTE_MODEL_ID, TEMPERATURE, TOP_P, REPEAT_PENALTY, MAX_LOOP_ITERATIONS,
       SOVEREIGN_DB_PATH, IS_WINDOWS, get_platform_shell, load_env!

# --- Platform Detection ---
const IS_WINDOWS = Sys.iswindows()

# --- Identity ---
const AGENT_NAME = "مجنون"
const AGENT_VERSION = "29.5-Julia"

# --- Mutable Configs (Loaded from Env/Config) ---
global WORKSPACE_ROOT = "c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/agent_workspace"
global LLM_PROVIDER = "openrouter"
global OPENROUTER_API_KEY = ""
global REMOTE_MODEL_ID = "google/gemini-2.5-flash"
global TEMPERATURE = 0.2
global TOP_P = 0.95
global REPEAT_PENALTY = 1.15
global MAX_LOOP_ITERATIONS = 50
global SOVEREIGN_DB_PATH = ""

function load_env!()
    # Read from .env if it exists
    env_path = joinpath(dirname(@__DIR__), ".env")
    if isfile(env_path)
        for line in eachline(env_path)
            line = strip(line)
            if isempty(line) || startswith(line, "#") || !occursin("=", line)
                continue
            end
            parts = split(line, "=", limit=2)
            k = strip(parts[1])
            v = strip(parts[2])
            ENV[k] = v
        end
    end

    # Set globals
    global WORKSPACE_ROOT = get(ENV, "WORKSPACE_DIR", joinpath(dirname(@__DIR__), "agent_workspace"))
    global OPENROUTER_API_KEY = get(ENV, "OPENROUTER_API_KEY", "")
    global LLM_PROVIDER = isempty(OPENROUTER_API_KEY) ? "local" : get(ENV, "LLM_PROVIDER", "openrouter")
    global REMOTE_MODEL_ID = get(ENV, "REMOTE_MODEL_ID", "google/gemini-2.5-flash")
    global TEMPERATURE = parse(Float64, get(ENV, "TEMPERATURE", "0.2"))
    global SOVEREIGN_DB_PATH = get(ENV, "SOVEREIGN_DB_PATH", joinpath(WORKSPACE_ROOT, ".memory", "sovereign_logic.db"))
    
    # Ensure workspaces directory exists
    mkpath(WORKSPACE_ROOT)
    mkpath(dirname(SOVEREIGN_DB_PATH))
end

function get_platform_shell()
    if IS_WINDOWS
        return ["powershell", "-ExecutionPolicy", "Bypass", "-Command"]
    else
        return ["bash", "-c"]
    end
end

end # module Constants
