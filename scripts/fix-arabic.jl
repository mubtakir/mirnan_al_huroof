#!/usr/bin/env julia
# -*- coding: utf-8 -*-
# fix-arabic.jl - A CLI tool to fix Arabic text rendering issues in terminal environments.

const RTL_EMBEDDING = "\u202B"
const POP_DIRECTIONAL = "\u202C"

const ARABIC_CHAR = "[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDDF\uFE70-\uFEFF]"
const ALLOWED_INTERMEDIATE = "[\u0020\u00090-9\u0660-\u0669\\-.,;:!?()'\"`\\[\\]{}،؛؟]"
const ARABIC_PHRASE_PATTERN = Regex("$ARABIC_CHAR+(?:$ALLOWED_INTERMEDIATE+$ARABIC_CHAR+)*")

function detect_bidi_support()
    if haskey(ENV, "WT_SESSION")
        return true
    end
    
    term_program = lowercase(get(ENV, "TERM_PROGRAM", ""))
    bidi_terminals = ["vscode", "apple_terminal", "iterm.app", "gnome-terminal", "konsole", "wezterm", "alacritty"]
    for t in bidi_terminals
        if occursin(t, term_program)
            return true
        end
    end
    
    term = lowercase(get(ENV, "TERM", ""))
    if occursin("xterm", term) || occursin("screen", term) || occursin("tmux", term)
        return true
    end
    
    if !Sys.iswindows()
        return true
    end
    
    return false
end

function fix_arabic(text::AbstractString, force::Bool, reverse_text::Bool)
    bidi_supported = detect_bidi_support()
    
    if bidi_supported && !force && !reverse_text
        return text
    end
    
    if reverse_text
        return replace(text, ARABIC_PHRASE_PATTERN => m -> reverse(m))
    end
    
    return replace(text, ARABIC_PHRASE_PATTERN => m -> RTL_EMBEDDING * m * POP_DIRECTIONAL)
end

function print_help()
    println("""
Fix Arabic text rendering in terminal/PowerShell (Julia Version).

Usage:
  julia fix-arabic.jl [options] [file]

Options:
  -f, --force      Force wrapping Arabic text with RTL control characters even if terminal seems to support it.
  -r, --reverse    Reverse Arabic characters (fallback for terminals with no Bidi support at all).
  -d, --detect     Detect and print whether the terminal supports native RTL/Bidi, then exit.
  -h, --help       Show this help message.

Examples:
  echo "مرحبا بكم" | julia fix-arabic.jl
  julia fix-arabic.jl -f input.txt
""")
end

function main()
    force = false
    reverse_text = false
    detect = false
    help = false
    filename = nothing

    for arg in ARGS
        if arg == "-f" || arg == "--force"
            force = true
        elseif arg == "-r" || arg == "--reverse"
            reverse_text = true
        elseif arg == "-d" || arg == "--detect"
            detect = true
        elseif arg == "-h" || arg == "--help"
            help = true
        elseif startswith(arg, "-")
            println(stderr, "Unknown argument: $arg")
            exit(1)
        else
            filename = arg
        end
    end

    if help
        print_help()
        exit(0)
    end

    if detect
        supported = detect_bidi_support()
        println("Terminal Bidi/RTL Support Detected: $supported")
        exit(0)
    end

    if !isnothing(filename)
        if !isfile(filename)
            println(stderr, "Error: File '$filename' not found.")
            exit(1)
        end
        open(filename, "r") do io
            for line in eachline(io, keep=true)
                print(fix_arabic(line, force, reverse_text))
            end
        end
    else
        # Process stdin
        for line in eachline(stdin, keep=true)
            print(fix_arabic(line, force, reverse_text))
            flush(stdout)
        end
    end
end

main()
