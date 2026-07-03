module BasilAgent

# 1. Load configuration and constants first
include("constants.jl")
using .Constants

# 2. Load core components
include("memory.jl")
using .Memory

include("rag_engine.jl")
using .RAGEngine

include("planner.jl")
using .Planner

include("tool_router.jl")
using .ToolRouter

# 3. Load basic tools
include("tools/file_system.jl")
include("tools/terminal_executor.jl")
include("tools/skill_creator.jl")
include("tools/browser.jl")
using .BrowserAgent
include("tools/docker_sandbox.jl")
using .DockerSandbox
include("tools/delegate_tool.jl")
using .DelegateTool

# 4. Load agent loop
include("agent.jl")
using .Agent

# 5. Load Flask-equivalent web app server
include("app.jl")
using .App

export Constants, Memory, RAGEngine, Planner, ToolRouter, Agent, App, MajnoonAgent, stream_run

function __init__()
    Constants.load_env!()
end

end # module BasilAgent
