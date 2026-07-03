module FileSystem

using Dates

export FileSystemManager, read_file, write_file, patch_file, list_dir, update_cwd!

mutable struct FileSystemManager
    workspace_root::String
    current_dir::String
end

function FileSystemManager(workspace_root::String)
    root = abspath(workspace_root)
    mkpath(root)
    return FileSystemManager(root, root)
end

function _get_safe_path(fs::FileSystemManager, file_path::String; allow_outside::Bool=true)::String
    clean_path = replace(file_path, '\\' => '/')
    clean_path = strip(clean_path)
    
    full_path = if !isabspath(clean_path)
        ws_name = basename(fs.workspace_root)
        if startswith(clean_path, "$ws_name/") || clean_path == ws_name
            normalized = strip(clean_path[length(ws_name)+1:end], '/')
            if !isempty(normalized) && !isdir(joinpath(fs.current_dir, clean_path))
                clean_path = normalized
            end
        end
        joinpath(fs.current_dir, clean_path)
    else
        clean_path
    end
    
    real_path = abspath(full_path)
    real_workspace = abspath(fs.workspace_root)
    
    if !allow_outside && !startswith(real_path, real_workspace)
        error("[SECURITY] 🛑 Access denied: Attempted to escape workspace ($real_path).")
    end
    
    return real_path
end

function read_file(fs::FileSystemManager, file_path::String, start_line::Union{Int, Nothing}=nothing, end_line::Union{Int, Nothing}=nothing)::String
    binary_extensions = Set([".png", ".jpg", ".jpeg", ".gif", ".ico", ".exe", ".dll", ".pyc", ".so", ".bin"])
    _, ext = splitext(lowercase(file_path))
    if ext in binary_extensions
        return "[SECURITY] 🛑 Access Denied: '$file_path' is a binary file. It cannot be read as text."
    end
    
    try
        path = _get_safe_path(fs, file_path)
        if !isfile(path)
            for test_ext in [".jl", ".py", ".js", ".md", ".json", ".txt", ".html", ".css"]
                if isfile(path * test_ext)
                    path = path * test_ext
                    file_path = file_path * test_ext
                    break
                end
            end
            if !isfile(path)
                return "[ERROR] File not found: $file_path. Use list_dir() to verify path."
            end
        end
        
        content_bytes = read(path)
        if 0x00 in content_bytes
            return "[SECURITY] 🛑 Access Denied: Binary content detected in '$file_path'."
        end
        
        content = String(content_bytes)
        lines = split(content, '\n')
        total_lines = length(lines)
        
        if start_line !== nothing || end_line !== nothing
            s = start_line !== nothing ? max(1, start_line) : 1
            e = end_line !== nothing ? min(total_lines, end_line) : total_lines
            
            selected = lines[s:e]
            content_numbered = join(["$(i+s-1): $line" for (i, line) in enumerate(selected)], "\n")
            return "--- [FILE: $file_path] Lines $s to $e of $total_lines ---\n$content_numbered"
        end
        
        return join(["$(i): $line" for (i, line) in enumerate(lines)], "\n")
    catch e
        return "[ERROR] Could not read file: $e"
    end
end

function write_file(fs::FileSystemManager, file_path::String, content::String)::String
    try
        path = _get_safe_path(fs, file_path)
        
        if isfile(path)
            checkpoint_dir = joinpath(fs.workspace_root, ".checkpoints")
            mkpath(checkpoint_dir)
            timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
            backup_name = "$(basename(file_path)).$(timestamp).bak"
            cp(path, joinpath(checkpoint_dir, backup_name), force=true)
        end
        
        mkpath(dirname(path))
        write(path, content)
        return "[SUCCESS] File saved successfully at $file_path (Backup created)"
    catch e
        return "[ERROR] Could not write file: $e"
    end
end

function patch_file(fs::FileSystemManager, file_path::String, old_block::String, new_block::String)::String
    if strip(old_block) == strip(new_block)
        return "[ERROR] old_block and new_block are identical! No actual change requested."
    end
    
    try
        path = _get_safe_path(fs, file_path)
        if !isfile(path)
            return "[ERROR] File not found: $file_path"
        end
        
        content = read(path, String)
        match_found = false
        
        if occursin(old_block, content)
            occurrences = length(collect(eachmatch(Regex(escape_string(old_block)), content)))
            if occurrences > 1
                return "[ERROR] Ambiguous replacement: block found $occurrences times. Add more context."
            end
            new_content = replace(content, old_block => new_block, count=1)
            write(path, new_content)
            return "[SUCCESS] Surgical patch applied to '$file_path'."
        end
        
        content_lines = split(content, '\n')
        block_lines = [strip(l) for l in split(old_block, '\n') if !isempty(strip(l))]
        
        if isempty(block_lines)
            return "[ERROR] old_block is empty."
        end
        
        for i in 1:(length(content_lines) - length(block_lines) + 1)
            if strip(content_lines[i]) == block_lines[1]
                potential_match = true
                for j in 1:length(block_lines)
                    if strip(content_lines[i+j-1]) != block_lines[j]
                        potential_match = false
                        break
                    end
                end
                if potential_match
                    exact_old_block = join(content_lines[i:(i+length(block_lines)-1)], "\n")
                    new_content = replace(content, exact_old_block => new_block, count=1)
                    write(path, new_content)
                    return "[SUCCESS] Surgical patch applied to '$file_path' (Fuzzy Match used: true)."
                end
            end
        end
        
        return "[ERROR] Code block not found in '$file_path'."
    catch e
        return "[ERROR] Patch failed: $e"
    end
end

function list_dir(fs::FileSystemManager, dir_path::String=".")::String
    try
        path = _get_safe_path(fs, dir_path)
        if !isdir(path)
            if isfile(path)
                return "[INFO] '$dir_path' is a FILE, not a directory. Use read_file('$dir_path') to view."
            end
            return "[ERROR] Not a directory: $dir_path"
        end
        
        items = readdir(path)
        header = "--- Listing for: $path ---\n"
        if isempty(items)
            return header * "[INFO] Directory is empty."
        end
        
        entries = String[]
        for item in sort(items)
            full = joinpath(path, item)
            if isdir(full)
                push!(entries, "📁 $item/")
            else
                size = filesize(full)
                size_str = if size < 1024
                    "$(size)B"
                elseif size < 1024 * 1024
                    "$(round(size/1024, digits=1))KB"
                else
                    "$(round(size/(1024*1024), digits=1))MB"
                end
                push!(entries, "📄 $item ($size_str)")
            end
        end
        return header * join(entries, "\n")
    catch e
        return "[ERROR] list_dir failed: $e"
    end
end

function update_cwd!(fs::FileSystemManager, new_relative_path::String)::String
    new_path = abspath(joinpath(fs.current_dir, new_relative_path))
    if isdir(new_path)
        fs.current_dir = new_path
        return "Changed directory to $(fs.current_dir)"
    end
    return "Error: Path $new_path not found"
end

end # module FileSystem
