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
Output:


10. Running `seq2seq` example using s3 datasets (from the root directory):
```bash
cd src/seq2seq
bash launch.sh
```
Output:




