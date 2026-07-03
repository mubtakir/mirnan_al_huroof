module DockerSandbox

export docker_check, docker_build, docker_run_code, docker_run_command, docker_list

const IMAGE_NAME = "basil-sovereign-sandbox:v1"

const DEFAULT_DOCKERFILE = """
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget build-essential nodejs npm \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    requests flask fastapi uvicorn \
    numpy pandas scipy matplotlib \
    pytest pytest-cov black flake8 mypy \
    httpx aiohttp
RUN useradd -m -u 1000 basil
WORKDIR /workspace
RUN chown basil:basil /workspace
USER basil
CMD ["/bin/bash"]
"""

function docker_check()::String
    out_buf = IOBuffer()
    err_buf = IOBuffer()
    try
        proc = run(pipeline(Cmd(["docker", "info", "--format", "{{.ServerVersion}}"]), stdout=out_buf, stderr=err_buf))
        version = strip(String(take!(out_buf)))
        return "[DOCKER SUCCESS] Docker daemon is available (Server version: $version)"
    catch e
        err = String(take!(err_buf))
        return "[DOCKER ERROR] Docker is not running or not installed in PATH.\nDetails: $e\nStderr: $err"
    end
end

function docker_build(workspace_root::String)::String
    sandbox_dir = joinpath(workspace_root, ".docker_sandbox")
    mkpath(sandbox_dir)
    dockerfile_path = joinpath(sandbox_dir, "Dockerfile")
    write(dockerfile_path, DEFAULT_DOCKERFILE)
    
    out_buf = IOBuffer()
    err_buf = IOBuffer()
    try
        proc = run(pipeline(Cmd(["docker", "build", "-t", IMAGE_NAME, sandbox_dir]), stdout=out_buf, stderr=err_buf))
        return "[DOCKER SUCCESS] Sandbox image '$IMAGE_NAME' built successfully."
    catch e
        err = String(take!(err_buf))
        return "[DOCKER ERROR] Failed to build sandbox image: $e\nStderr: $err"
    end
end

function docker_run_code(workspace_root::String, code::String)::String
    tmp_dir = mktempdir(prefix="basil_docker_")
    code_path = joinpath(tmp_dir, "run.py")
    write(code_path, code)
    
    # We want to mount the code folder and run it
    # Windows paths need forward slashes or double backslashes in docker mounts
    clean_tmp_dir = replace(tmp_dir, '\\' => '/')
    
    cmd = Cmd([
        "docker", "run",
        "--rm",
        "--network=none",
        "--memory=256m",
        "--cpus=1.0",
        "--read-only",
        "--tmpfs=/tmp:size=64m",
        "-v", "$clean_tmp_dir:/code:ro",
        IMAGE_NAME,
        "python", "/code/run.py"
    ])
    
    out_buf = IOBuffer()
    err_buf = IOBuffer()
    try
        run(pipeline(cmd, stdout=out_buf, stderr=err_buf))
        out_str = String(take!(out_buf))
        err_str = String(take!(err_buf))
        result = "[DOCKER RUN SUCCESS]\nSTDOUT:\n" * out_str
        if !isempty(err_str)
            result *= "\nSTDERR:\n" * err_str
        end
        return result
    catch e
        out_str = String(take!(out_buf))
        err_str = String(take!(err_buf))
        return "[DOCKER RUN FAIL] execution crashed: $e\nStdout: $out_str\nStderr: $err_str"
    finally
        rm(tmp_dir, recursive=true, force=true)
    end
end

function docker_run_command(workspace_root::String, command::String)::String
    cmd = Cmd([
        "docker", "run",
        "--rm",
        "--network=none",
        "--memory=256m",
        IMAGE_NAME,
        "/bin/bash", "-c", command
    ])
    out_buf = IOBuffer()
    err_buf = IOBuffer()
    try
        run(pipeline(cmd, stdout=out_buf, stderr=err_buf))
        out_str = String(take!(out_buf))
        err_str = String(take!(err_buf))
        result = "[DOCKER RUN SUCCESS]\nSTDOUT:\n" * out_str
        if !isempty(err_str)
            result *= "\nSTDERR:\n" * err_str
        end
        return result
    catch e
        out_str = String(take!(out_buf))
        err_str = String(take!(err_buf))
        return "[DOCKER RUN FAIL] execution crashed: $e\nStdout: $out_str\nStderr: $err_str"
    end
end

function docker_list()::String
    out_buf = IOBuffer()
    err_buf = IOBuffer()
    try
        run(pipeline(Cmd(["docker", "ps", "--format", "table {{.Names}}\\t{{.Image}}\\t{{.Status}}"]), stdout=out_buf, stderr=err_buf))
        return "[DOCKER CONTAINERS]\n" * String(take!(out_buf))
    catch e
        return "[DOCKER ERROR] Failed to list containers: $e"
    end
end

end # module DockerSandbox
