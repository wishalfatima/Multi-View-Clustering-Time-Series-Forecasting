import numpy as np
from ViewGeneration.view_generation import process_m3_dataset  # M3 dataset preprocessing
from ViewGeneration.view_generation import process_time_series  # Generate views
from ViewEncoding.m3_preprocessing import process_m3_dataset_enc # M3 dataset preprocessing for encoding
from ViewEncoding.feature_extraction import extract_time_series_features, extract_image_features  # Feature extraction
from sklearn.preprocessing import MinMaxScaler
from ViewSelection.view_selection import select_views  # View selection
from FuzzyClustering.demo import fuzzy_multi_view_clustering  # Multi-view clustering
from Forecasting.m3_forecasting import run_experiments  # Run forecasting experiments (UNCOMMENTED)
from Forecasting.ensemble import median_ensemble_evaluation  # Median ensemble evaluation (UNCOMMENTED)

min_max_scaler = MinMaxScaler()

if __name__ == "__main__":
    """
    Unified pipeline:
    1. Load & process M3 dataset
    2. Generate images (RP, GAF, MTF) from time series
    3.1. Extract numerical features from time series
    3.2. Extract CNN features from images
    4. Select best views using diversity & similarity
    5. Perform multi-view fuzzy clustering
    6. Run multiple experiments & collect predictions
    7. Evaluate ensemble predictions using Median approach
    """

    # Step 1: Process M3 dataset
    # Use LOCAL path instead of hardcoded workspace path
    file_path = "./data/M3Month.csv"
    data = process_m3_dataset(file_path)   

    # Step 2: Generate images from time series
    # Use LOCAL paths for all image outputs

    # #  micro (commented out - uncomment if needed)
    # micro = data['MICRO']
    # image_output_path = "./data/generated_images"
    # process_time_series(micro, f'{image_output_path}/MICRO', 18, 0)

    # #  industry (commented out - uncomment if needed)
    # industry = data['INDUSTRY']
    # image_output_path = "./data/generated_images"
    # process_time_series(industry, f'{image_output_path}/INDUSTRY', 18, True, 474)

    # #  macro (commented out - uncomment if needed)
    # macro = data['MACRO']
    # image_output_path = "./data/generated_images"
    # process_time_series(macro, f'{image_output_path}/MACRO', 18, True, 808)

    # #  finance (commented out - uncomment if needed)
    # finance = data['FINANCE']
    # image_output_path = "./data/generated_images"
    # process_time_series(finance, f'{image_output_path}/FINANCE', 18, True, 1120)

    #  demo (active)
    demo = data['DEMOGRAPHIC']
    image_output_path = "./data/generated_images"
    process_time_series(demo, f'{image_output_path}/DEMOGRAPHIC', 18, True, 1265)

    # #  other (commented out - uncomment if needed)
    # other = data['OTHER']
    # image_output_path = "./data/generated_images"
    # process_time_series(other, f'{image_output_path}/OTHER', 18, True, 1376)

    # Step 3: Extract features using Autoencoder
    # Use the same local file_path
    processed_data = process_m3_dataset_enc(file_path)
    features_list = []
    if processed_data:
        print("Time-series processing complete. Categories:", list(processed_data.keys()))

    # Step 3.1: Extract numerical features only for 'DEMOGRAPHIC'
    demo_ts_features_dict = {
        label: extract_time_series_features(data)
        for label, data in processed_data.items()
        if label == "DEMOGRAPHIC"
    }
    demo_ts_features = list(demo_ts_features_dict.values())[0]
    features_list.append(min_max_scaler.fit_transform(demo_ts_features))
    # np.save("demo_time_series_features.npy", demo_ts_features)
    print("Time-series feature extraction complete.")

    # Step 3.2: Extract image-based CNN features only for 'DEMOGRAPHIC'
    demo_rp_img_features_dict = {
        label: extract_image_features(data, f"{image_output_path}/{label}/rp", gray=True)
        for label, data in processed_data.items()
        if label == "DEMOGRAPHIC"
    }
    demo_rp_img_features = list(demo_rp_img_features_dict.values())[0]
    features_list.append(min_max_scaler.fit_transform(demo_rp_img_features))

    
    demo_mtf_img_features_dict = {
        label: extract_image_features(data, f"{image_output_path}/{label}/mtf", gray=False)
        for label, data in processed_data.items()
        if label == "DEMOGRAPHIC"
    }
    demo_mtf_img_features = list(demo_mtf_img_features_dict.values())[0]
    features_list.append(min_max_scaler.fit_transform(demo_mtf_img_features))

    
    demo_gasf_img_features_dict = {
        label: extract_image_features(data, f"{image_output_path}/{label}/gasf", gray=False)
        for label, data in processed_data.items()
        if label == "DEMOGRAPHIC"
    }
    demo_gasf_img_features = list(demo_gasf_img_features_dict.values())[0]
    features_list.append(min_max_scaler.fit_transform(demo_gasf_img_features))

    
    demo_gadf_img_features_dict = {
        label: extract_image_features(data, f"{image_output_path}/{label}/gadf", gray=False)
        for label, data in processed_data.items()
        if label == "DEMOGRAPHIC"
    }
    demo_gadf_img_features = list(demo_gadf_img_features_dict.values())[0]
    features_list.append(min_max_scaler.fit_transform(demo_gadf_img_features))

    # np.save("demo_image_features.npy", features_list)
    print("Image feature extraction complete.")

    # Save features to LOCAL path
    np.save('./data/features/demo_All_features.npy', features_list)
    print("All features saved for DEMOGRAPHIC.")

    # Step 4: Select best views
    selected_views = {}
    for label in processed_data.keys():
        selected_views[label] = select_views(features_list, threshold=0.1)

    np.save("selected_views.npy", selected_views)
    print("View selection complete. Selected views saved.")

    # Step 5: Perform multi-view fuzzy clustering
    print("Running multi-view fuzzy clustering...")
    clustering_results = fuzzy_multi_view_clustering(dataset="demo")
    np.save("clustering_results.npy", clustering_results)
    print("Clustering completed. Results saved.")

    # Step 6: Run multiple experiments and collect predictions (UNCOMMENTED)
    print("Running experiments...")
    combined_predictions, TestY, TrainY = run_experiments(n_runs=2, ae_name="Autoencoder_Model", dim=16)
    np.save("combined_predictions.npy", combined_predictions)
    print("Experiments completed. Predictions saved.")

    # Step 7: Perform Median Ensemble Evaluation (UNCOMMENTED)
    print("Performing Median Ensemble Evaluation...")
    dataset_name = "M3_Monthly"
    mase_freq = 12  # Monthly data

    ensemble_results = median_ensemble_evaluation(
        combined_predictions=combined_predictions,
        TestY=TestY,
        TrainY=TrainY,
        dataset_name=dataset_name,
        mase_freq=mase_freq,
        save_path="ensemble_results"
        )

    print("Ensemble evaluation completed. Results saved.")
    print("Final Ensemble Results:", ensemble_results)
    
    print("\n" + "="*50)
    print("✅ Full pipeline completed successfully including forecasting!")
    print("📊 Check your results:")
    print("   - Features saved in: ./data/features/demo_All_features.npy")
    print("   - Selected views: selected_views.npy")
    print("   - Clustering results: clustering_results.npy")
    print("   - Predictions: combined_predictions.npy")
    print("   - Ensemble results: ensemble_results/")
    print("="*50)