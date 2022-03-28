
## KITTI detection

### Convert to COCO format

```bash
pip install fiftyone

INPUT_DIR=$(fiftyone zoo datasets find kitti --split validation)
OUTPUT_DIR=/tmp/fiftyone/kitti-coco

fiftyone convert \
    --input-dir ${INPUT_DIR} --input-type fiftyone.types.FiftyOneImageDetectionDataset \
    --output-dir ${OUTPUT_DIR} --output-type fiftyone.types.COCODetectionDataset
```
