include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const IRP = MirnanNew.Physics.IntentResponsePlanner

@testset "intent response planner" begin
    how_plan = IRP.detect_response_intent("كيف يتعلم الإنسان")
    @test how_plan.intent == "mechanism"
    @test how_plan.subject == "الإنسان"
    @test how_plan.action == "يتعلم"
    how_answer = IRP.render_planned_response(how_plan)
    @test how_answer == ""
    @test !occursin("فهم السياق", how_answer)
    @test !occursin("تنظيم الخطوات", how_answer)
    @test IRP.has_plannable_response(how_plan)
    how_profile = IRP.intent_gravity_profile(how_plan)
    @test how_profile.intent == "mechanism"
    @test "الإنسان" in how_profile.guidance_terms
    @test "يتعلم" in how_profile.guidance_terms
    @test "بخير" in how_profile.repulsion_terms

    heart_plan = IRP.detect_response_intent("كيف يعمل القلب؟")
    @test heart_plan.intent == "mechanism"
    @test heart_plan.subject == "القلب"
    @test heart_plan.action == "يعمل"

    weather_plan = IRP.detect_response_intent("كيف هو الطقس اليوم؟")
    @test weather_plan.intent == "descriptive"
    @test occursin("الطقس", weather_plan.subject)
    weather_profile = IRP.intent_gravity_profile(weather_plan)
    @test weather_profile.intent == "descriptive"
    @test "بخير" in weather_profile.repulsion_terms

    conditional_plan = IRP.detect_response_intent("إذا زاد العلم زاد الفهم")
    @test conditional_plan.intent == "conditional"
    @test occursin("زاد العلم", conditional_plan.cause)
    @test occursin("زاد الفهم", conditional_plan.result)
    conditional_answer = IRP.render_planned_response(conditional_plan)
    @test occursin("إذا زاد العلم", conditional_answer)
    @test occursin("زاد الفهم", conditional_answer)

    why_plan = IRP.detect_response_intent("لماذا يزيد العلم الفهم")
    @test why_plan.intent == "causal"
    @test !IRP.has_plannable_response(why_plan)
    why_answer = IRP.render_planned_response(why_plan; related_terms=["المعرفة", "الإدراك"])
    @test why_answer == ""
    why_profile = IRP.intent_gravity_profile(why_plan)
    @test IRP.has_gravity_profile(why_profile)
    @test why_profile.intent == "causal"
    @test why_profile.syntax_multiplier > 1.0
    @test "العلم" in why_profile.guidance_terms
    @test "الفهم" in why_profile.guidance_terms
    @test "رايك" in why_profile.repulsion_terms

    opinion_plan = IRP.detect_response_intent("ما رأيك في القراءة والكتابة؟")
    @test opinion_plan.intent == "opinion"
    @test occursin("القراءة", opinion_plan.subject)
    opinion_answer = IRP.render_planned_response(opinion_plan; related_terms=["المعرفة", "الفهم"])
    @test occursin("أرى أن", opinion_answer)
    @test occursin("القراءة", opinion_answer)
    @test occursin("نافع", opinion_answer)

    useful_plan = IRP.detect_response_intent("هل العلم مفيد دائما؟")
    @test useful_plan.intent == "opinion"
    @test occursin("العلم", useful_plan.subject)

    duty_plan = IRP.detect_response_intent("هل التعلم واجب على الجميع؟")
    @test duty_plan.intent == "opinion"
    @test occursin("التعلم", duty_plan.subject)

    greeting_plan = IRP.detect_response_intent("السلام عليكم")
    @test greeting_plan.intent == "dialogue"
    @test !IRP.has_plannable_response(greeting_plan)
    greeting_profile = IRP.intent_gravity_profile(greeting_plan)
    @test IRP.has_gravity_profile(greeting_profile)
    @test greeting_profile.question_charge < 0
    @test greeting_profile.response_charge > 0
    @test greeting_profile.syntax_multiplier > 1.0
    @test "السلام" in greeting_profile.guidance_terms
    @test "هل" in greeting_profile.repulsion_terms
    @test IRP.render_planned_response(greeting_plan) == ""

    farewell_plan = IRP.detect_response_intent("إلى اللقاء")
    @test farewell_plan.intent == "dialogue"
    farewell_profile = IRP.intent_gravity_profile(farewell_plan)
    @test "اللقاء" in farewell_profile.guidance_terms
    @test "سعيدا" in farewell_profile.guidance_terms

    wellbeing_plan = IRP.detect_response_intent("كيف حالك؟")
    @test wellbeing_plan.intent == "dialogue"
    wellbeing_profile = IRP.intent_gravity_profile(wellbeing_plan)
    @test "حالي" in wellbeing_profile.guidance_terms
    @test "بخير" in wellbeing_profile.guidance_terms

    learning_plan = IRP.detect_response_intent("هل تحب التعلم؟")
    @test learning_plan.intent == "dialogue"
    learning_profile = IRP.intent_gravity_profile(learning_plan)
    @test "أحب" in learning_profile.guidance_terms
    @test "التعلم" in learning_profile.guidance_terms

    knowledge_plan = IRP.detect_response_intent("هل تؤمن بالمعرفة؟")
    @test knowledge_plan.intent == "dialogue"
    knowledge_profile = IRP.intent_gravity_profile(knowledge_plan)
    @test "أؤمن" in knowledge_profile.guidance_terms
    @test "المعرفة" in knowledge_profile.guidance_terms

    beauty_plan = IRP.detect_response_intent("ما أجمل شيء في الحياة؟")
    @test beauty_plan.intent == "dialogue"
    @test "أجمل" in IRP.intent_gravity_profile(beauty_plan).guidance_terms

    day_plan = IRP.detect_response_intent("كيف تقضي يومك؟")
    @test day_plan.intent == "dialogue"
    @test "يومي" in IRP.intent_gravity_profile(day_plan).guidance_terms

    dialogue_opinion_plan = IRP.detect_response_intent("ما هو رأيك في القراءة؟")
    @test dialogue_opinion_plan.intent == "opinion"
    @test occursin("القراءة", dialogue_opinion_plan.subject)
end
