cd #!/usr/bin/env sh
set -e

python /SegNet_RiceCulm/test_segmentation_RiceCulm.py --model /SegNet_RiceCulm/SegNet_RiceCulm_deploy.prototxt --weights /SegNet_RiceCulm/test_weights_RiceCulm.caffemodel --data="/SegNet_RiceCulm/Sample/test.txt"
