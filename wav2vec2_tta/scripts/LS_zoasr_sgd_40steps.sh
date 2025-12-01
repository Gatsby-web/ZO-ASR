#LS + 0
python main_zoasr.py --asr facebook/wav2vec2-base-960h \
                        --steps 40 \
                        --dataset_name librispeech \
                        --dataset_dir path/to/dataset \
                        --temp 2.5 \
                        --episodic \
                        --em_coef 0.3 \
                        --reweight \
                        --log_dir path/to/logs \
                        --lr 2e-5 \
                        --non_blank \
                        --train_feature \
                        --extra_noise 0 \
                        --opt zo \
                        --q_rge 8 \

#LS + 0.005
python main_zoasr.py --asr facebook/wav2vec2-base-960h \
                        --steps 40 \
                        --dataset_name librispeech \
                        --dataset_dir path/to/dataset \
                        --temp 2.5 \
                        --episodic \
                        --em_coef 0.3 \
                        --reweight \
                        --log_dir path/to/logs \
                        --lr 2e-5 \
                        --non_blank \
                        --train_feature \
                        --extra_noise 0.005 \
                        --opt zo \
                        --q_rge 8 \

#LS + 0.01
python main_zoasr.py --asr facebook/wav2vec2-base-960h \
                        --steps 40 \
                        --dataset_name librispeech \
                        --dataset_dir path/to/dataset \
                        --temp 2.5 \
                        --episodic \
                        --em_coef 0.3 \
                        --reweight \
                        --log_dir path/to/logs \
                        --lr 2e-5 \
                        --non_blank \
                        --train_feature \
                        --extra_noise 0.01 \
                        --opt zo \
                        --q_rge 8 \