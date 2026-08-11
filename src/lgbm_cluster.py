from collections import defaultdict

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from metrics import predict_mean_baseline, print_report, regression_report


def load_data(npz_path):
    data = np.load(npz_path)
    return data['features'], data['labels'], data['filenames']

def group_and_rank_features(train_path, val_path, distance_threshold=0.3):
    print("="*60)
    print("LIGHTGBM FEATURE DISTILLATION & CLUSTERING")
    print("="*60)

    # 1. Load Extracted CNN Features
    print("\n[PHASE 1] Loading extracted features...")
    X_train, y_train, _ = load_data(train_path)
    X_val, y_val, _ = load_data(val_path)
    
    print(f"  -> Train shape: {X_train.shape} | Val shape: {X_val.shape}")

    # 2. Hierarchical Feature Clustering
    print("\n[PHASE 2] Calculating Spearman correlations and clustering features...")
    # Calculate correlation matrix of the FEATURES (transpose X_train)
    # Using Spearman because neural network features are often non-linear
    corr_matrix, _ = spearmanr(X_train, axis=0)
    
    # Ensure symmetry and clip rounding errors
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    
    # Convert correlation to a distance metric (1 - |correlation|)
    # Features that are highly correlated (positive or negative) will have distance near 0
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # Perform Ward hierarchical clustering
    linkage_matrix = hierarchy.linkage(squareform(distance_matrix), method='ward')
    
    # Cut the dendrogram to form flat clusters
    # distance_threshold=0.3 means features with >0.7 correlation are grouped together
    cluster_labels = hierarchy.fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
    
    num_clusters = len(np.unique(cluster_labels))
    print(f"  -> Reduced 256 raw features into {num_clusters} distinct 'Feature Families'.")

    # 3. LightGBM Training
    print("\n[PHASE 3] Training LightGBM Regressor...")
    # We train on all 256 raw features; LightGBM will naturally ignore the useless ones
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    print(f"  -> Best LightGBM Iteration: {model.best_iteration_}")

    val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
    print_report("LightGBM (val set)", regression_report(y_val, val_pred))
    print_report("predict-the-mean floor (val set)", predict_mean_baseline(y_train, y_val))

    # 4. Aggregate Importance by Cluster
    print("\n[PHASE 4] Aggregating Feature Importance by Family...")
    # 'gain' measures how much a feature improved the magnitude prediction when it was used to split a tree
    raw_importances = model.booster_.feature_importance(importance_type='gain')
    
    family_importances = defaultdict(float)
    family_members = defaultdict(list)
    
    for feature_idx, (cluster_id, importance) in enumerate(zip(cluster_labels, raw_importances)):
        family_importances[cluster_id] += importance
        family_members[cluster_id].append(feature_idx)
        
    # Sort families by their total importance
    sorted_families = sorted(family_importances.items(), key=lambda x: x[1], reverse=True)
    
    # Print the Top 10 Feature Families
    print("\n--- TOP 10 MOST IMPORTANT FEATURE FAMILIES ---")
    for rank, (cluster_id, total_gain) in enumerate(sorted_families[:10]):
        members = family_members[cluster_id]
        print(f"{rank+1}. Family #{cluster_id:<3} | Total Gain: {total_gain:,.2f} | CNN Feature IDs: {members}")

    return model, cluster_labels, sorted_families

if __name__ == "__main__":
    
    # Ensure you have run the PyTorch extraction script first!
    TRAIN_NPZ = "extracted_features/train_features_60s.npz"
    VAL_NPZ = "extracted_features/val_features_60s.npz"
    
    # You can tweak the distance_threshold to get more or fewer families
    group_and_rank_features(TRAIN_NPZ, VAL_NPZ, distance_threshold=0.3)
