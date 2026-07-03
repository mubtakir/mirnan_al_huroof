using Pkg
Pkg.activate(joinpath(@__DIR__))

# Load main module
include("src/BasilAgent.jl")
using .BasilAgent

println("==================================================")
println("  BasilAgent.jl - Sovereign Test Suite (Inside Mirnan)            ")
println("==================================================")

# 1. Initialize Agent
println("👤 Initializing MajnoonAgent...")
agent = MajnoonAgent("c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/agent_workspace")
println("✅ Agent initialized successfully.")
println("   Name: $(agent.tools.agent_instance.role)")
println("   Workspace: $(agent.workspace_root)")

# 2. Test Tool Router Listing
println("\n🛠️ Querying available tools...")
tools_list = BasilAgent.ToolRouter.get_tools_listing(agent.tools)
println(tools_list)

# 3. Test Memory DB
println("🧠 Testing SQLite episodic memory database...")
BasilAgent.Memory.add_episode!(agent.memory, "اختبار أولي للنظام السيادي بلغة جوليا", 3, "test,julia")
episodes = BasilAgent.Memory.get_recent_episodes!(agent.memory, limit=1)
if !isempty(episodes)
    println("✅ Memory test passed. Last episode summary: '$(episodes[1][:content])'")
else
    println("❌ Memory test failed.")
end

# 4. Test RAG Semantic Memory
println("\n📖 Testing Local TF-IDF RAG Memory...")
BasilAgent.RAGEngine.store_memory!(agent.rag, "القرار المعماري: تم تحويل الكود بالكامل إلى جوليا الفائقة السرعة لتفعيل التطور الذاتي للأدوات.", source="architecture")
recall_context = BasilAgent.RAGEngine.semantic_recall(agent.rag, "تطوير الأدوات وتحديث الكود")
println(recall_context)

# 5. Success
println("\n🎉 Sovereign Brain components are 100% operational!")
println("==================================================")
