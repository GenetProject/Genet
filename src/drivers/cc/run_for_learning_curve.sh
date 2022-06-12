python src/simulator/evaluate_training_models_synthetic.py \
    --dataset-dir data/cc/learning_curve \
    --save-dir results/cc/learning_curve/ \
    --cc pretrained \
    --models-path models/cc/learning_curve/pretrained/seed_20

for seed in 10 20 30; do
    python src/simulator/evaluate_training_models_synthetic.py \
        --dataset-dir data/cc/learning_curve \
        --save-dir results/cc/learning_curve/ \
        --cc genet_bbr_old \
        --models-path models/cc/learning_curve/genet_bbr_old/seed_${seed}

    python src/simulator/evaluate_training_models_synthetic.py \
        --dataset-dir data/cc/learning_curve \
        --save-dir results/cc/learning_curve/ \
        --cc cl1 \
        --models-path models/cc/learning_curve/cl1/seed_${seed}

    python src/simulator/evaluate_training_models_synthetic.py \
        --dataset-dir data/cc/learning_curve \
        --save-dir results/cc/learning_curve/ \
        --cc cl2 \
        --models-path models/cc/learning_curve/cl2/seed_${seed}

    python src/simulator/evaluate_training_models_synthetic.py \
        --dataset-dir data/cc/learning_curve \
        --save-dir results/cc/learning_curve/ \
        --cc cl3 \
        --models-path models/cc/learning_curve/cl3/seed_${seed}

    python src/simulator/evaluate_training_models_synthetic.py \
        --dataset-dir data/cc/learning_curve \
        --save-dir results/cc/learning_curve/ \
        --cc udr3 \
        --models-path models/cc/learning_curve/udr3/seed_${seed}
done
