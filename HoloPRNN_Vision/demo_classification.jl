"""
demo_classification.jl
Demonstrates the PhasNet CHL classifier on simple 8x8 geometric shapes.
"""

using Pkg
Pkg.activate(@__DIR__)

const DEMO_DIR = @__DIR__

include(joinpath(DEMO_DIR, "src", "HoloPRNN_Vision.jl"))
using .HoloPRNN_Vision

function run_demo()
    println("=========================================================================")
    println("      HoloPRNN Vision - Classifier Demo (CHL Classification)")
    println("=========================================================================")

    pat1 = zeros(64)
    for y in 0:7, x in 0:3
        pat1[y * 8 + x + 1] = 1.0
    end

    pat2 = zeros(64)
    for y in 0:7, x in 4:7
        pat2[y * 8 + x + 1] = 1.0
    end

    pat3 = zeros(64)
    for y in 0:7, x in 0:7
        if (y + x) % 2 == 0
            pat3[y * 8 + x + 1] = 1.0
        end
    end

    X = [pat1, pat2, pat3]
    Y = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    class_names = ["Left Half", "Right Half", "Checkerboard"]

    println("Initializing PhasNetClassifier (64 -> 32 -> 3)...")
    clf = PhasNetClassifier(64, 32, 3)

    println("Training with Contrastive Hebbian Learning (CHL) for 200 epochs...")
    for epoch in 1:200
        train_classifier!(clf, X, Y; epochs=1, lr=0.015)
        if any(isnan, clf.K1) || any(isnan, clf.K2)
            println("  NaN detected at epoch $(epoch).")
            break
        elseif epoch % 50 == 0
            loss = sum(compute_classification_loss(clf, img_flat, target_class) for (img_flat, target_class) in zip(X, Y))
            println("  Epoch $epoch | Loss: $(round(loss, digits=4))")
        end
    end

    println("Training complete.")
    println("-" ^ 70)
    println("Testing classifier on the trained patterns:")

    predictions = predict_batch(clf, X)
    success = true
    for i in 1:3
        pred_idx = predictions[i]
        if pred_idx != i
            success = false
        end
        status = pred_idx == i ? "SUCCESS" : "FAILED"
        println("  $status | Pattern $i: '$(class_names[i])' -> Predicted Class: $pred_idx ('$(class_names[pred_idx])')")
    end

    println("-" ^ 70)
    println(success ? "SUCCESS: all training patterns classified correctly." : "Classifier did not converge on all shapes. Adjust parameters.")
    println("=========================================================================")
end

run_demo()
