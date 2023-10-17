result=1
echo "Running Intermediate Fusion"
while [ $result -ne 0 ]; do
    CUDA_VISIBLE_DEVICES=0 python -m asg.scripts.scene_generator \
    --sg_hypes_yaml asg/hypes_yaml/sg_genetic_algorithm.yaml \
    --model_dir /path/to/opencood/logs/model_dir \
    --fusion_method intermediate \
    --v2x v2x \
    --client_port 2000 \
    --load_opencda_format_data
    result=$?
    sleep 5
done