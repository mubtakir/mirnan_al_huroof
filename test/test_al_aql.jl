using Test
using MirnanNew

function run_test()
    println("==========================================================")
    println("     المحاكاة الفيزيائية للديناميكا الطورية لـ (العقل)     ")
    println("     المندمجة كأداة في مكتبة مرنان الجديد (MirnanNew)     ")
    println("      (الهيكلة البرمجية القائمة على الكلاسات والدوال)      ")
    println("==========================================================")
    
    # ─── إنشاء فضاء المحاكاة ───
    space = SimulationSpace()
    
    # ─── السيناريو الأول: الأسد والغزال ───
    println("\n🦁 السيناريو الأول: الأسد زأر فخاف الغزال فجرى")
    println("----------------------------------------------------------")
    
    # إنشاء الكيانات (كلاسات برمجية ملموسة)
    أسد = Lion("الأسد", 2.0)
    set_property!(أسد, "energy", 0.8, 0.0) # الأسد لديه طاقة عالية
    
    غزال = Gazelle("الغزال", 0.6)
    set_property!(غزال, "fear", 0.1, 0.0)      # خوف منخفض جداً بالبداية
    set_property!(غزال, "awareness", 0.7, 0.0) # انتباه ووعي عالٍ
    set_property!(غزال, "motion", 0.1, 0.0)    # ساكن تقريباً
    
    # تسجيل الكيانات
    register_entity!(space, أسد)
    register_entity!(space, غزال)
    
    println("الحالة الابتدائية للغزال:")
    print_entity(غزال)
    
    # تفاعل: الأسد زأر على الغزال (تمثيل برمجى مباشر لدالة الفعل)
    interact!(space, "الأسد", roar!, "الغزال")
    
    println("\nالحالة النهائية للغزال:")
    print_entity(غزال)
    
    # تحقق من صحة سيناريو الغزال
    fear_amp, _ = get_property(غزال, "fear")
    motion_amp, _ = get_property(غزال, "motion")
    @test fear_amp ≈ 0.1
    @test motion_amp ≈ 0.8
    
    # ─── السيناريو الثاني: الأرض والأثاث ───
    println("\n🏢 السيناريو الثاني: الأرض اهتزت فسقط الأثاث")
    println("----------------------------------------------------------")
    
    # إنشاء الكيانات
    أرض = Earth("الأرض", 10.0)
    set_property!(أرض, "energy", 0.9, 0.0) # زلزال ذو طاقة عالية
    
    أثاث = Furniture("الأثاث", 1.2)
    set_property!(أثاث, "stability", 0.95, 0.0) # مستقر جداً بالبداية
    set_property!(أثاث, "integrity", 0.9, 0.0)  # سليم وتماسك ممتاز
    set_property!(أثاث, "motion", 0.0, 0.0)     # ساكن تماماً
    
    register_entity!(space, أرض)
    register_entity!(space, أثاث)
    
    println("الحالة الابتدائية للأثاث:")
    print_entity(أثاث)
    
    # تفاعل: الأرض اهتزت على الأثاث (تمثيل برمجى مباشر لدالة الفعل)
    interact!(space, "الأرض", shake!, "الأثاث")
    
    println("\nالحالة النهائية للأثاث:")
    print_entity(أثاث)
    
    # تحقق من صحة سيناريو الأثاث
    stability_amp, _ = get_property(أثاث, "stability")
    motion_amp_f, _ = get_property(أثاث, "motion")
    integrity_amp, _ = get_property(أثاث, "integrity")
    @test stability_amp ≈ 0.0
    @test motion_amp_f ≈ 0.7
    @test integrity_amp ≈ 0.9 * 0.6
    
    # ─── طباعة سجل أحداث المحاكاة ───
    println("\n📜 سجل الأحداث والترابطات السببية المستنتجة:")
    println("==========================================================")
    for log_line in space.log
        println(log_line)
    end
    println("==========================================================")
end

run_test()
