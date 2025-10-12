
import shutil

videos = [
    'data_video5_20250909110956225_2025091310250004',
    'data_video6_20250826142203247_2025082814264202',
    'data_video7_20250909110956225',
    'data_video11_20250818110226366',
    'data_video12_20250819134002235',
]

for video in videos:
    prefix = video.split('_')[1]
    shutil.copy(f"{video}/adagaus_density_curve/adagaus_density_analysis.png", f"daily_summary/1012/{prefix}_adagaus_density_analysis.png")
    shutil.copy(f"{video}/burr_density_curve/burr_density_analysis.png", f"daily_summary/1012/{prefix}_burr_density_analysis.png")
    shutil.copy(f"{video}/coil_wear_analysis/visualizations/temporal_trends.png", f"daily_summary/1012/{prefix}_coil_wear_temporal_trends.png")
    shutil.copy(f"{video}/spot_temporal_curve/spot_temporal_analysis_smoothed.png", f"daily_summary/1012/{prefix}_spot_temporal_analysis_smoothed.png")
    shutil.copy(f"{video}/white_patch_test/method_comparison.png", f"daily_summary/1012/{prefix}_method_comparison.png")
    shutil.copy(f"{video}/white_patch_test/temporal_curves_4x8.png", f"daily_summary/1012/{prefix}_white_patch_tempral_curves_4x8.png")