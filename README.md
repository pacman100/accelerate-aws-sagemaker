# accelerate-aws-sagemaker
Examples showcasing AWS SageMaker integration of 🤗 Accelerate. Just give the `accelerate config` and do `accelerate launch` 🚀. As simple as that!

1. Set up the accelerate config by running `accelerate config --config_file accelerate_config.yaml` and answer the SageMaker questions and set it up.

2. Below is a sample config which is using aws `profile` to launch training job using 🤗 SageMaker estimator. It also has the `iam_role_name` which has the needed SageMaker permissions specified. In this config it is replaced `xxxxx` as user needs to specify it based on their corresponding AWS setup.

```yaml
base_job_name: accelerate-sagemaker-1
compute_environment: AMAZON_SAGEMAKER
distributed_type: DATA_PARALLEL
ec2_instance_type: ml.p3.16xlarge
iam_role_name: xxxxx
image_uri: null
mixed_precision: fp16
num_machines: 1
profile: xxxxx
py_version: py38
pytorch_version: 1.10.2
region: us-east-1
transformers_version: 4.17.0
use_cpu: false
```
3. One can specify a custom docker image instead of Official 🤗 DLCs through the accelerate config questionnaire. When this isn't provided, the latest Official 🤗 DLC will be used.

4. Support for input channels pointing to S3 data locations via TSV file, e.g., below are the contents of sagemaker_inputs.tsv whose location is given as part of accelerate config setup.
```tsv
channel_name	data_location
train	s3://sagemaker-sample/samples/datasets/imdb/train
test	s3://sagemaker-sample/samples/datasets/imdb/test
```

6. Support for SageMaker metrics logging via TSV file, e.g., below are the contents of the sagemaker_metrics_definition.tsv whose location is given as part of accelerate config setup.
```tsv
metric_name	metric_regex
accuracy	'accuracy': ([0-9.]+)
f1	'f1': ([0-9.]+)
```

7. Example of accelerate config with above features setup [XXXXX values are AWS account specific]:
```yaml
base_job_name: accelerate-sagemaker-1
compute_environment: AMAZON_SAGEMAKER
distributed_type: DATA_PARALLEL
ec2_instance_type: ml.p3.16xlarge
iam_role_name: XXXXX
image_uri: 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.8.1-transformers4.10.2-gpu-py36-cu111-ubuntu18.04
mixed_precision: fp16
num_machines: 1
profile: XXXXX
py_version: py38
pytorch_version: 1.10.2
region: us-east-1
sagemaker_inputs_file: sagemaker_inputs.tsv
sagemaker_metrics_file: sagemaker_metrics_definition.tsv
transformers_version: 4.17.0
use_cpu: false
```
8. Put `requirements.txt` with all the needed libraries for running the training script.

9. Running `text-classification` example using s3 datasets (from the root directory):
```bash
cd src/text-classification
bash launch.sh
```
The contents of launch.sh
```bash
accelerate launch  --config_file accelerate_config.yaml train_using_s3_data.py \
    --mixed_precision "fp16"
```
Output logs:
```bash
...

[1,mpirank:0,algo-1]<stdout>:algo-1:79:1300 [0] NCCL INFO Launch mode Parallel
[1,mpirank:0,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:3,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:1,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:2,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:6,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:5,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:7,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:4,algo-1]<stderr>:INFO:root:Reducer buckets have been rebuilt in this iteration.
[1,mpirank:0,algo-1]<stdout>:epoch 0: {'accuracy': 0.6838235294117647, 'f1': 0.8122270742358079}
[1,mpirank:0,algo-1]<stdout>:epoch 1: {'accuracy': 0.7205882352941176, 'f1': 0.8256880733944955}
[1,mpirank:0,algo-1]<stdout>:epoch 2: {'accuracy': 0.75, 'f1': 0.838095238095238}
2022-09-21 13:21:05,187 sagemaker-training-toolkit INFO     Waiting for the process to finish and give a return code.
2022-09-21 13:21:05,188 sagemaker-training-toolkit INFO     Done waiting for a return code. Received 0 from exiting process.
2022-09-21 13:21:05,188 sagemaker-training-toolkit INFO     Reporting training SUCCESS
```


10. Running `seq2seq` example:
```bash
cd src/seq2seq
bash launch.sh
```
The contents of launch.sh
```bash
accelerate launch --config_file accelerate_config.yaml run_seq2seq_no_trainer.py \
    --dataset_name "smangrul/MuDoConv" \
    --max_source_length 128 \
    --source_prefix "chatbot: " \
    --max_target_length 64 \
    --val_max_target_length 64 \
    --val_min_target_length 20 \
    --n_val_batch_generations 5 \
    --n_train 10000 \
    --n_val 1000 \
    --pad_to_max_length True\
    --num_beams 10 \
    --model_name_or_path "facebook/blenderbot-400M-distill" \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-6 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 100 \
    --output_dir "/opt/ml/model" \
    --seed 25 \
    --logging_steps 100 \
    --report_name "blenderbot_400M_finetuning"
```
Output logs:
```
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:37:39 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: smddp
[1,mpirank:0,algo-1]<stderr>:Num processes: 8
[1,mpirank:0,algo-1]<stderr>:Process index: 0
[1,mpirank:0,algo-1]<stderr>:Local process index: 0
[1,mpirank:0,algo-1]<stderr>:Device: cuda:0
[1,mpirank:0,algo-1]<stderr>:
[1,mpirank:7,algo-1]<stderr>:09/21/2022 13:37:39 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: smddp
[1,mpirank:7,algo-1]<stderr>:Num processes: 8
[1,mpirank:7,algo-1]<stderr>:Process index: 7
[1,mpirank:7,algo-1]<stderr>:Local process index: 7
[1,mpirank:7,algo-1]<stderr>:Device: cuda:7
[1,mpirank:7,algo-1]<stderr>:
[1,mpirank:3,algo-1]<stderr>:09/21/2022 13:37:39 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: smddp
[1,mpirank:3,algo-1]<stderr>:Num processes: 8
[1,mpirank:3,algo-1]<stderr>:Process index: 3
[1,mpirank:3,algo-1]<stderr>:Local process index: 3
[1,mpirank:3,algo-1]<stderr>:Device: cuda:3
[1,mpirank:3,algo-1]<stderr>:
[1,mpirank:1,algo-1]<stderr>:09/21/2022 13:37:39 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: smddp
[1,mpirank:1,algo-1]<stderr>:Num processes: 8
[1,mpirank:1,algo-1]<stderr>:Process index: 1
[1,mpirank:1,algo-1]<stderr>:Local process index: 1
[1,mpirank:1,algo-1]<stderr>:Device: cuda:1
[1,mpirank:1,algo-1]<stderr>:
[1,mpirank:4,algo-1]<stderr>:09/21/2022 13:37:39 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: smddp
[1,mpirank:4,algo-1]<stderr>:Num processes: 8
[1,mpirank:4,algo-1]<stderr>:Process index: 4
[1,mpirank:4,algo-1]<stderr>:Local process index: 4
[1,mpirank:4,algo-1]<stderr>:Device: cuda:4
[1,mpirank:4,algo-1]<stderr>:
[1,mpirank:6,algo-1]<stderr>:09/21/2022 13:37:39 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: smddp
[1,mpirank:6,algo-1]<stderr>:Num processes: 8
[1,mpirank:6,algo-1]<stderr>:Process index: 6
[1,mpirank:6,algo-1]<stderr>:Local process index: 6
[1,mpirank:6,algo-1]<stderr>:Device: cuda:6
[1,mpirank:6,algo-1]<stderr>:
[1,mpirank:0,algo-1]<stdout>:algo-1:728:1533 [0] NCCL INFO Launch mode Parallel
[1,mpirank:5,algo-1]<stderr>:09/21/2022 13:37:39 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: smddp
[1,mpirank:5,algo-1]<stderr>:Num processes: 8
[1,mpirank:5,algo-1]<stderr>:Process index: 5
[1,mpirank:5,algo-1]<stderr>:Local process index: 5
[1,mpirank:5,algo-1]<stderr>:Device: cuda:5
[1,mpirank:5,algo-1]<stderr>:
[1,mpirank:2,algo-1]<stderr>:09/21/2022 13:37:39 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: smddp
[1,mpirank:2,algo-1]<stderr>:Num processes: 8
[1,mpirank:2,algo-1]<stderr>:Process index: 2
[1,mpirank:2,algo-1]<stderr>:Local process index: 2
[1,mpirank:2,algo-1]<stderr>:Device: cuda:2

...

[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ - ***** Running training *****
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ -   Num examples = 10000
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ -   Num Epochs = 1
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ -   Instantaneous batch size per device = 16
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 128
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ -   Gradient Accumulation steps = 1
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:31 - INFO - __main__ -   Total optimization steps = 79
[1,mpirank:0,algo-1]<stderr>:#015  0%|          | 0/79 [00:00<?, ?it/s]
[1,mpirank:2,algo-1]<stderr>:#015Downloading builder script:   0%|          | 0.00/2.85k [00:00<?, ?B/s]
[1,mpirank:2,algo-1]<stderr>:#015Downloading builder script: 7.65kB [00:00, 3.92MB/s]
[1,mpirank:6,algo-1]<stderr>:#015Downloading builder script:   0%|          | 0.00/2.85k [00:00<?, ?B/s]
[1,mpirank:6,algo-1]<stderr>:#015Downloading builder script: 7.65kB [00:00, 3.35MB/s]
[1,mpirank:1,algo-1]<stderr>:#015Downloading builder script:   0%|          | 0.00/2.85k [00:00<?, ?B/s]
[1,mpirank:1,algo-1]<stderr>:#015Downloading builder script: 7.65kB [00:00, 4.45MB/s]
[1,mpirank:3,algo-1]<stdout>:[2022-09-21 13:38:31.653: W smdistributed/modelparallel/torch/nn/predefined_hooks.py:48] Found unsupported HuggingFace version 4.22.1 for automated tensor parallelism. HuggingFace modules will not be automatically distributed. You can use smp.tp_register_with_module API to register desired modules for tensor parallelism, or directly instantiate an smp.nn.DistributedModule. Supported HuggingFace transformers versions for automated tensor parallelism: ['4.17.0']
[1,mpirank:0,algo-1]<stdout>:[2022-09-21 13:38:31.700: W smdistributed/modelparallel/torch/nn/predefined_hooks.py:48] Found unsupported HuggingFace version 4.22.1 for automated tensor parallelism. HuggingFace modules will not be automatically distributed. You can use smp.tp_register_with_module API to register desired modules for tensor parallelism, or directly instantiate an smp.nn.DistributedModule. Supported HuggingFace transformers versions for automated tensor parallelism: ['4.17.0']
[1,mpirank:2,algo-1]<stdout>:[2022-09-21 13:38:31.724: W smdistributed/modelparallel/torch/nn/predefined_hooks.py:48] Found unsupported HuggingFace version 4.22.1 for automated tensor parallelism. HuggingFace modules will not be automatically distributed. You can use smp.tp_register_with_module API to register desired modules for tensor parallelism, or directly instantiate an smp.nn.DistributedModule. Supported HuggingFace transformers versions for automated tensor parallelism: ['4.17.0']
[1,mpirank:0,algo-1]<stderr>:You're using a BlenderbotTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
[1,mpirank:4,algo-1]<stdout>:[2022-09-21 13:38:31.839: W smdistributed/modelparallel/torch/nn/predefined_hooks.py:48] Found unsupported HuggingFace version 4.22.1 for automated tensor parallelism. HuggingFace modules will not be automatically distributed. You can use smp.tp_register_with_module API to register desired modules for tensor parallelism, or directly instantiate an smp.nn.DistributedModule. Supported HuggingFace transformers versions for automated tensor parallelism: ['4.17.0']
[1,mpirank:5,algo-1]<stdout>:[2022-09-21 13:38:31.840: W smdistributed/modelparallel/torch/nn/predefined_hooks.py:48] Found unsupported HuggingFace version 4.22.1 for automated tensor parallelism. HuggingFace modules will not be automatically distributed. You can use smp.tp_register_with_module API to register desired modules for tensor parallelism, or directly instantiate an smp.nn.DistributedModule. Supported HuggingFace transformers versions for automated tensor parallelism: ['4.17.0']
[1,mpirank:7,algo-1]<stdout>:[2022-09-21 13:38:31.841: W smdistributed/modelparallel/torch/nn/predefined_hooks.py:48] Found unsupported HuggingFace version 4.22.1 for automated tensor parallelism. HuggingFace modules will not be automatically distributed. You can use smp.tp_register_with_module API to register desired modules for tensor parallelism, or directly instantiate an smp.nn.DistributedModule. Supported HuggingFace transformers versions for automated tensor parallelism: ['4.17.0']
[1,mpirank:6,algo-1]<stdout>:[2022-09-21 13:38:31.842: W smdistributed/modelparallel/torch/nn/predefined_hooks.py:48] Found unsupported HuggingFace version 4.22.1 for automated tensor parallelism. HuggingFace modules will not be automatically distributed. You can use smp.tp_register_with_module API to register desired modules for tensor parallelism, or directly instantiate an smp.nn.DistributedModule. Supported HuggingFace transformers versions for automated tensor parallelism: ['4.17.0']
[1,mpirank:1,algo-1]<stdout>:[2022-09-21 13:38:31.850: W smdistributed/modelparallel/torch/nn/predefined_hooks.py:48] Found unsupported HuggingFace version 4.22.1 for automated tensor parallelism. HuggingFace modules will not be automatically distributed. You can use smp.tp_register_with_module API to register desired modules for tensor parallelism, or directly instantiate an smp.nn.DistributedModule. Supported HuggingFace transformers versions for automated tensor parallelism: ['4.17.0']
[1,mpirank:0,algo-1]<stderr>:#015  1%|▏         | 1/79 [00:01<02:13,  1.71s/it]
[1,mpirank:7,algo-1]<stderr>:09/21/2022 13:38:33 - INFO - root - Reducer buckets have been rebuilt in this iteration.
[1,mpirank:4,algo-1]<stderr>:09/21/2022 13:38:33 - INFO - root - Reducer buckets have been rebuilt in this iteration.
[1,mpirank:5,algo-1]<stderr>:09/21/2022 13:38:33 - INFO - root - Reducer buckets have been rebuilt in this iteration.
[1,mpirank:2,algo-1]<stderr>:09/21/2022 13:38:33 - INFO - root - Reducer buckets have been rebuilt in this iteration.
[1,mpirank:3,algo-1]<stderr>:09/21/2022 13:38:33 - INFO - root - Reducer buckets have been rebuilt in this iteration.
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:33 - INFO - root - Reducer buckets have been rebuilt in this iteration.
[1,mpirank:1,algo-1]<stderr>:09/21/2022 13:38:33 - INFO - root - Reducer buckets have been rebuilt in this iteration.
[1,mpirank:6,algo-1]<stderr>:09/21/2022 13:38:33 - INFO - root - Reducer buckets have been rebuilt in this iteration.
[1,mpirank:0,algo-1]<stderr>:#015  3%|▎         | 2/79 [00:02<01:11,  1.07it/s]
...

[1,mpirank:0,algo-1]<stderr>:#015100%|██████████| 79/79 [00:19<00:00,  4.79it/s]
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:50 - INFO - __main__ - Epoch 0 training took 19.50162172317505 seconds
[1,mpirank:0,algo-1]<stdout>:starting evaluation
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:55 - INFO - __main__ - printing few sample generations and corresponding labels from eval set
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:55 - INFO - __main__ - prompt | generated | label
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:55 - INFO - __main__ - chatbot:your persona: i work as an electrician. i always sleep 8 hours a day. </s>  <s> Which level are you at?</s> |  I'm at the top of the ladder. I work for an electrical company. |  I received on-the-job training when i first started
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:55 - INFO - __main__ - chatbot:your persona: i work as an electrician. i always sleep 8 hours a day. </s>  <s> Which level are you at? </s> <s> I received on-the-job training when i first started </s> <s> Thats great! How long have you been doing this work? </s> |  I've been working as an Electrician for about 5 years now. It's a great job. |  For a good number of years now.
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:55 - INFO - __main__ - chatbot:your persona: i work as an electrician. i always sleep 8 hours a day. </s>  <s> Thats great! How long have you been doing this work? </s> |  I've been working as an Electrician for about 5 years now. I love it! |  For a good number of years now.
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:38:55 - INFO - __main__ - chatbot:your persona: i work as an electrician. i always sleep 8 hours a day. </s>  <s> Thats great! How long have you been doing this work?  </s> <s> For a good number of years now. </s> <s> Sounds like it could be a dangerous job at times.</s> |  It can be at times, but I enjoy it. What do you do for a living? |  That it is,especially if you dont take the proper measures

... 

[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:39:02 - INFO - __main__ - chatbot:your persona: i'm a painter and love to create art. i'm a talented singer and have won several competitions. </s>  <s> I am doing ok. Thanks for asking. Playing guitar helps me cope. My father has a cool collection of guitars.</s> |  That's cool. I wish I could play an instrument. Do you play any other instruments? |  Nice, any particular songs you normally play?
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:39:02 - INFO - __main__ - chatbot:your persona: i'm a painter and love to create art. i'm a talented singer and have won several competitions. </s>  <s> I love to play Jimi Hendrix songs. Are you a fan of 1960's rock? What is your favorite genre?</s> |  I am a big fan of the 60's. My favorite band is the Beatles. |  Of course, I love 60's rock, my favorite has to be classic rock for sure. Zeppelin and The Doors have to be my favorite bands
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:39:02 - INFO - __main__ - chatbot:your persona: i'm a painter and love to create art. i'm a talented singer and have won several competitions. </s>  <s> I love the Doors! They have such a unique sound. Do you have a favorite Doors song?</s> |  My favorite song of theirs is "When I Was Your Man". What's yours? |  It's a tie between People are strange and Love me two times. What's your favorite?
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:39:02 - INFO - __main__ - chatbot:your persona: i'm a painter and love to create art. i'm a talented singer and have won several competitions. </s>  <s> I think my favorite is Love Street. It has such a haunting melody. Have you heard that one?</s> |  No, I haven't. I'll have to check it out. What genre is it? |  Yeah, it's a pretty great song, Jim Morrison was just an amazing songwriter, I aspire to make songs as good as his one day.

...

[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:39:17 - INFO - __main__ - {'bleu': 1.7067114414104911}
[1,mpirank:0,algo-1]<stdout>:evaluation completed
[1,mpirank:0,algo-1]<stderr>:09/21/2022 13:39:17 - INFO - __main__ - Epoch 0 evaluation took 24.294514417648315 seconds
[1,mpirank:0,algo-1]<stderr>:Configuration saved in /opt/ml/model/config.json
[1,mpirank:0,algo-1]<stderr>:Model weights saved in /opt/ml/model/pytorch_model.bin
[1,mpirank:0,algo-1]<stderr>:tokenizer config file saved in /opt/ml/model/tokenizer_config.json
[1,mpirank:0,algo-1]<stderr>:Special tokens file saved in /opt/ml/model/special_tokens_map.json
[1,mpirank:0,algo-1]<stderr>:#015100%|██████████| 79/79 [00:47<00:00,  1.65it/s]
2022-09-21 13:39:27,753 sagemaker-training-toolkit INFO     Waiting for the process to finish and give a return code.
2022-09-21 13:39:27,753 sagemaker-training-toolkit INFO     Done waiting for a return code. Received 0 from exiting process.
2022-09-21 13:39:27,754 sagemaker-training-toolkit INFO     Reporting training SUCCESS
```


