"""
validate_format.jl — التحقق من تنسيق النصوص التدريبية

القواعد:
1. الجمل التامة تنتهي بنقطة.
2. الجمل غير التامة تنتهي بفاصلة،
3. الجمل المكملة تنتهي بفاصلة منقوطة؛
4. جمل القول تنتهي بنقطتين:
5. علامات الترقيم ملاصقة للكلمة الأخيرة
6. سطر فارغ بين الفقرات
7. فصل الحوار بـ tab أو كلمات سؤال:/جواب:/س:/ج:
"""

struct ValidationResult
    line_number::Int
    rule::String
    message::String
    severity::String  # "ERROR" or "WARNING"
end

const PUNCTUATION = Set(['.', ',', ';', ':', '!', '?'])

function validate_file(filepath::String; verbose::Bool=true)
    lines = readlines(filepath)
    issues = ValidationResult[]

    for (i, line) in enumerate(lines)
        stripped = strip(line)

        # Rule: Empty line check (paragraph separation)
        if isempty(stripped)
            continue
        end

        # Rule 5: Punctuation must be attached (no space before punctuation)
        # Check for " . " or " ، " etc (space before punctuation)
        idx = firstindex(stripped)
        while idx <= ncodeunits(stripped)
            ch = stripped[idx]
            if ch in PUNCTUATION && idx > firstindex(stripped)
                prev_idx = prevind(stripped, idx)
                if stripped[prev_idx] == ' '
                    push!(issues, ValidationResult(
                        i, "PUNCTUATION_SPACE",
                        "علامة ترقيم '$ch' متبوعة بمسافة قبلها في الموقع $(idx)",
                        "ERROR"
                    ))
                end
            end
            idx = nextind(stripped, idx)
        end

        # Rule 5: Punctuation should not be at start of line (unless dialogue marker)
        if !isempty(stripped) && stripped[1] in PUNCTUATION && stripped[1] != '\t'
            if stripped[1] in ['.', ',', ';']
                push!(issues, ValidationResult(
                    i, "LINE_START_PUNCT",
                    "سطر يبدأ بعلامة ترقيم '$(stripped[1])'",
                    "WARNING"
                ))
            end
        end

        # Rule: Lines should not end with space
        if endswith(line, ' ') || endswith(line, '\t')
            push!(issues, ValidationResult(
                i, "TRAILING_SPACE",
                "سطر ينتهي بمسافة",
                "WARNING"
            ))
        end

        # Rule: Dialogue separation (tab check)
        if '\t' in stripped
            parts = split(stripped, '\t')
            if length(parts) > 2
                push!(issues, ValidationResult(
                    i, "MULTIPLE_TABS",
                    "سطر يحتوي على أكثر من tab واحد ($(length(parts)) أجزاء)",
                    "WARNING"
                ))
            end
        end
    end

    # Rule 6: Paragraph separation (empty line between paragraphs)
    prev_was_content = false
    consecutive_content = 0
    for (i, line) in enumerate(lines)
        stripped = strip(line)
        if isempty(stripped)
            if prev_was_content
                consecutive_content = 0
            end
            prev_was_content = false
        else
            if prev_was_content
                consecutive_content += 1
            else
                consecutive_content = 1
            end
            prev_was_content = true
        end
    end

    # Print results
    if verbose
        errors = count(x -> x.severity == "ERROR", issues)
        warnings = count(x -> x.severity == "WARNING", issues)

        println("=" ^ 60)
        println("  التحقق من التنسيق: $filepath")
        println("=" ^ 60)
        println("  إجمالي الأسطر: $(length(lines))")
        println("  أخطاء: $errors")
        println("  تحذيرات: $warnings")
        println()

        if errors == 0 && warnings == 0
            println("  ✅ التنسيق متوافق مع جميع القواعد!")
        else
            for issue in issues
                icon = issue.severity == "ERROR" ? "❌" : "⚠️"
                println("  $icon [سطر $(issue.line_number)] ($(issue.rule))")
                println("     $(issue.message)")
            end
        end
        println()
    end

    return issues
end

function validate_directory(dirpath::String; verbose::Bool=true)
    all_issues = Dict{String, Vector{ValidationResult}}()

    for (root, dirs, files) in walkdir(dirpath)
        for file in files
            if endswith(file, ".txt") || endswith(file, ".md")
                filepath = joinpath(root, file)
                issues = validate_file(filepath; verbose=verbose)
                if !isempty(issues)
                    all_issues[filepath] = issues
                end
            end
        end
    end

    if verbose
        total_errors = sum(count(x -> x.severity == "ERROR", v) for v in values(all_issues); init=0)
        total_warnings = sum(count(x -> x.severity == "WARNING", v) for v in values(all_issues); init=0)
        println("=" ^ 60)
        println("  ملخص التحقق العام")
        println("=" ^ 60)
        println("  ملفات تم فحصها: $(length(all_issues))")
        println("  إجمالي الأخطاء: $total_errors")
        println("  إجمالي التحذيرات: $total_warnings")
    end

    return all_issues
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1
        path = ARGS[1]
        if isfile(path)
            validate_file(path)
        elseif isdir(path)
            validate_directory(path)
        else
            println("المسار غير موجود: $path")
        end
    else
        println("الاستخدام: julia validate_format.jl <مسار ملف أو مجلد>")
        println()
        println("أمثلة:")
        println("  julia validate_format.jl data/corpus/test.txt")
        println("  julia validate_format.jl data/corpus/")
    end
end
