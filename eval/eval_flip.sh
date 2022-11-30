# coffee-martini
VIDEO_DIR=/data/machine/TensoRFVideo/log/ddr_hmq_coffe_schedule1_star_25000_small_coffee_martini_2.0/imgs_test_all
GT=../data/coffee_martini/frames_2/cam00/
python eval/main.py --output ${VIDEO_DIR}/_comp_video.mp4 \
                    --gt $GT \
                    --downsample 2 \
                    --flip_path ${VIDEO_DIR}/flips/ \
                    --interval 1
ffmpeg -f image2 -r 30 -i ${VIDEO_DIR}/flips/%05d.png ${VIDEO_DIR}/flip.mp4

VIDEO_DIR=/data/machine/TensoRFVideo/log/ddr_hmq_flame_25000_small_flame_salmon_2.0/None/imgs_test_all
GT=../data/flame_salmon/frames_2/cam00/
python eval/main.py --output ${VIDEO_DIR}/_comp_video.mp4 \
                    --gt $GT \
                    --downsample 2 \
                    --flip_path ${VIDEO_DIR}/flips/ \
                    --interval 1
ffmpeg -f image2 -r 30 -i ${VIDEO_DIR}/flips/%05d.png ${VIDEO_DIR}/flip.mp4

VIDEO_DIR=/data/machine/TensoRFVideo/log/ddr_hmq_highmotion2_beef_lr2_ks5_25000_small_cut_roasted_beef_2.0/tensorf_flower_VM_coffe_small3_static_dynamic12/imgs_test_all
GT=../data/cut_roasted_beef/frames_2/cam00/
python eval/main.py --output ${VIDEO_DIR}/_comp_video.mp4 \
                    --gt $GT \
                    --downsample 2 \
                    --flip_path ${VIDEO_DIR}/flips/ \
                    --interval 1
ffmpeg -f image2 -r 30 -i ${VIDEO_DIR}/flips/%05d.png ${VIDEO_DIR}/flip.mp4

VIDEO_DIR=/data/machine/TensoRFVideo/log/ddr_hmq_highmotion2_steak_lr2_ks5_25000_small_flame_steak_2.0/imgs_test_all
GT=../data/flame_steak/frames_2/cam00/
python eval/main.py --output ${VIDEO_DIR}/_comp_video.mp4 \
                    --gt $GT \
                    --downsample 2 \
                    --flip_path ${VIDEO_DIR}/flips/ \
                    --interval 1
ffmpeg -f image2 -r 30 -i ${VIDEO_DIR}/flips/%05d.png ${VIDEO_DIR}/flip.mp4

VIDEO_DIR=/data/machine/TensoRFVideo/log/ddr_hmq_cook_spinch3_temvar2_star3_25000_small_cook_spinach_2.0/imgs_test_all
GT=../data/cook_spinach/frames_2/cam00/
python eval/main.py --output ${VIDEO_DIR}/_comp_video.mp4 \
                    --gt $GT \
                    --downsample 2 \
                    --flip_path ${VIDEO_DIR}/flips/ \
                    --interval 1
ffmpeg -f image2 -r 30 -i ${VIDEO_DIR}/flips/%05d.png ${VIDEO_DIR}/flip.mp4

VIDEO_DIR=/data/machine/TensoRFVideo/log/ddr_hmq_highmotion2_sear_lr2_ks5_25000_small_sear_steak_2.0/imgs_test_all
GT=../data/sear_steak/frames_2/cam00/
python eval/main.py --output ${VIDEO_DIR}/_comp_video.mp4 \
                    --gt $GT \
                    --downsample 2 \
                    --flip_path ${VIDEO_DIR}/flips/ \
                    --interval 1
ffmpeg -f image2 -r 30 -i ${VIDEO_DIR}/flips/%05d.png ${VIDEO_DIR}/flip.mp4
