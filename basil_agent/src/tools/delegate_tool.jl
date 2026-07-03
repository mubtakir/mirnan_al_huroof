module DelegateTool

export delegate_task

function delegate_task(parent_agent::Any, goal::String, context::Union{String, Nothing}=nothing, role::String="Engineer")::String
    try
        # Get parent agent details
        workspace_root = parent_agent.workspace_root
        
        # Create a clean child agent in the same workspace root
        child = Main.BasilAgent.Agent.MajnoonAgent(workspace_root)
        child.role = role
        
        # Setup specific role constraints
        full_goal = """
        [تفويض مهمة لوكيل فرعي: $role]
        الهدف: $goal
        السياق الإضافي: $(context === nothing ? "لا يوجد" : context)
        تعليمات هامة: قم بحل المهمة بصفة سيادية ودون اعتذار. قدّم نتائج وحلول واضحة ومباشرة.
        """
        
        results = String[]
        Main.BasilAgent.Agent.stream_run(child, full_goal) do event
            if event["type"] == "thought"
                # Keep thoughts to see reasoning
                push!(results, "💭 " * event["content"])
            elseif event["type"] == "system" && occursin("Error", event["content"])
                push!(results, "❌ " * event["content"])
            end
        end
        
        summary = join(results, "\n")
        return "[SUB-AGENT $(uppercase(role)) SUCCESS]\n" * first(summary, 2500)
    catch e
        return "[SUB-AGENT $(uppercase(role)) ERROR] Failed delegation: $e"
    end
end

end # module DelegateTool
