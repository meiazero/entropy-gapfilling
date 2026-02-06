=== File ===
metadata: /opt/datasets/satellite-images/metadata.parquet

=== Dimensions ===
rows: 77916
columns: 16
memory (approx): 73.40 MiB

=== Columns ===
patch_id | source_file | acquisition_date | col_off | row_off | crs | transform | bands | clean_path | mask_synthetic_path | degraded_inf_path | degraded_40_path | degraded_30_path | degraded_20_path | split | satellite

=== Dtypes ===
patch_id int64
col_off int64
row_off int64
source_file object
acquisition_date object
crs object
transform object
bands object
clean_path object
mask_synthetic_path object
degraded_inf_path object
degraded_40_path object
degraded_30_path object
degraded_20_path object
split object
satellite object

=== Head (5) ===
patch_id source_file acquisition_date col_off row_off crs transform bands clean_path mask_synthetic_path degraded_inf_path degraded_40_path degraded_30_path degraded_20_path split satellite
1 2025-08-30.tif 2025-08-30 0 0 EPSG:4326 | 0.00, 0.00,-38.60|\n| 0.00,-0.00,-5.53|\n| 0.00, 0.00, 1.00| B2;B3;B4;B8 sentinel2/clean/0000001.tif sentinel2/masks/synthetic/0000001.tif sentinel2/degraded/inf_db/0000001.tif sentinel2/degraded/40_db/0000001.tif sentinel2/degraded/30_db/0000001.tif sentinel2/degraded/20_db/0000001.tif val sentinel2
2 2025-08-30.tif 2025-08-30 32 0 EPSG:4326 | 0.00, 0.00,-38.60|\n| 0.00,-0.00,-5.53|\n| 0.00, 0.00, 1.00| B2;B3;B4;B8 sentinel2/clean/0000002.tif sentinel2/masks/synthetic/0000002.tif sentinel2/degraded/inf_db/0000002.tif sentinel2/degraded/40_db/0000002.tif sentinel2/degraded/30_db/0000002.tif sentinel2/degraded/20_db/0000002.tif val sentinel2
3 2025-08-30.tif 2025-08-30 64 0 EPSG:4326 | 0.00, 0.00,-38.60|\n| 0.00,-0.00,-5.53|\n| 0.00, 0.00, 1.00| B2;B3;B4;B8 sentinel2/clean/0000003.tif sentinel2/masks/synthetic/0000003.tif sentinel2/degraded/inf_db/0000003.tif sentinel2/degraded/40_db/0000003.tif sentinel2/degraded/30_db/0000003.tif sentinel2/degraded/20_db/0000003.tif val sentinel2
4 2025-08-30.tif 2025-08-30 96 0 EPSG:4326 | 0.00, 0.00,-38.60|\n| 0.00,-0.00,-5.53|\n| 0.00, 0.00, 1.00| B2;B3;B4;B8 sentinel2/clean/0000004.tif sentinel2/masks/synthetic/0000004.tif sentinel2/degraded/inf_db/0000004.tif sentinel2/degraded/40_db/0000004.tif sentinel2/degraded/30_db/0000004.tif sentinel2/degraded/20_db/0000004.tif val sentinel2
5 2025-08-30.tif 2025-08-30 128 0 EPSG:4326 | 0.00, 0.00,-38.59|\n| 0.00,-0.00,-5.53|\n| 0.00, 0.00, 1.00| B2;B3;B4;B8 sentinel2/clean/0000005.tif sentinel2/masks/synthetic/0000005.tif sentinel2/degraded/inf_db/0000005.tif sentinel2/degraded/40_db/0000005.tif sentinel2/degraded/30_db/0000005.tif sentinel2/degraded/20_db/0000005.tif val sentinel2

=== Data Quality ===
duplicate rows (all columns): 0
rows with at least 1 null: 0

=== Masks ===
mask_synthetic_path encodes valid pixels as 1 and gaps as 0. Degraded rasters have NaN at gap locations (mask==0); clean rasters keep full signal.

=== Dates - acquisition_date ===
valid: 77916 / 77916
min: 2023-10-10 00:00:00+00:00
max: 2025-09-29 00:00:00+00:00

=== Distribution - satellite (top 20) ===
satellite
sentinel2 73984
landsat9 1936
landsat8 1936
modis 60

=== Distribution - split (top 20) ===
split
train 52842
test 14849
val 10225
