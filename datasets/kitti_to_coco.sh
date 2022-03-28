# Kitti dataset format
# <dataset_dir>/
#    data/
#        <uuid1>.<ext>
#        <uuid2>.<ext>
#        ...
#    labels/
#        <uuid1>.txt
#        <uuid2>.txt
#        ...

INPUT_DIR=/path/to/dataset/
OUTPUT_DIR=./kitti-coco

fiftyone convert \
    --input-dir ${INPUT_DIR} --input-type fiftyone.types.KITTIDetectionDataset \
    --output-dir ${OUTPUT_DIR} --output-type fiftyone.types.COCODetectionDataset
