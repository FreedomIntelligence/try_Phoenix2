# try_Phoenix2
Phoenix2 code in dev

## Dependency
module load cuda11.8/toolkit/11.8.0

pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip uninstall -y ninja && pip install ninja
pip install flash-attn --no-build-isolation

pip install -r requirements.txt tokenizers sentencepiece

## Structure

- TinyLlama_FSDP: 原来的预训练代码
- TinyLlama_deepspeed_check1： 利用Deepspeed策略
    - pip install deepspeed
    - 文档:
        - 代码: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html
        - 参数: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DeepSpeedStrategy.html#lightning.pytorch.strategies.DeepSpeedStrategy
    - 改动: 
        - 更换策略: pretrain.py line 10, 95
        - 为了用checkpointing技术: pretrain.py line 141
        - scripts/convert_zero_checkpoint.py Zero3 checkpoint转换
- TinyLlama_deepspeed_check2： 利用Deepspeed策略
    - 改动: 
        - 为了用checkpointing技术: lit-gpt/model/gpt.foward line 69-119
- TinyLlama_collosal： 
    - lightening collosal 需要torch<2.0 放弃


## Usage

```
sbatch multinode_pretrain.sh
```

### 数据
```
python scripts/prepare_starcoder.py --source_path /path/to/starcoderdata/ --tokenizer_path data/llama --destination_path data/slim_star_combined --split train --percentage 1.0
python scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path data/llama  --destination_path data/slim_star_combined --split validation --percentage 1.0
python scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path data/llama  --destination_path data/slim_star_combined --split train --percentage 1.0
```

### 多节点训练

```
bash 
```


## Calculate index for Strategy choosing
```
python pre_train_math.py
```
```
-----------Model_Size and GPU_Mem-----------
+--------------+------------------------+----------------------+
| Model size/B | ratio(NHIDDEN/NLAYERS) | Usable_mem_per_GPU/G |
+--------------+------------------------+----------------------+
|     1.18     |           93           |          79          |
+--------------+------------------------+----------------------+
-----------With Mixed Precision(bp16)-----------
-----Memory_reference_indicator(Batch_size=1)-----
+-------------------------+----------+------------------+-------------------+
| Module                  |   Size/B |   Eval_memory/GB |   Train_momery/GB |
+=========================+==========+==================+===================+
| emb                     |     0.07 |             0.14 |              1.12 |
+-------------------------+----------+------------------+-------------------+
| one_layer               |     0.05 |             0.1  |              0.81 |
+-------------------------+----------+------------------+-------------------+
| input                   |     0    |             0.01 |              0.01 |
+-------------------------+----------+------------------+-------------------+
| activation(batchsize=1) |     1.77 |             3.54 |              3.54 |
+-------------------------+----------+------------------+-------------------+
| ALL                     |     2.95 |             5.91 |             22.39 |
+-------------------------+----------+------------------+-------------------+
-----Strategy_reference_indicator(Batch_size=1)-----
+------------+--------------------------+---------------------------+
| Strategy   |   Eval_memory_per_gpu/GB |   Train_momery_per_gpu/GB |
+============+==========================+===========================+
| Zero1      |                     2.35 |                      8.44 |
+------------+--------------------------+---------------------------+
| Zero2      |                     2.35 |                      6.11 |
+------------+--------------------------+---------------------------+
| Zero3      |                     0.03 |                      3.79 |
+------------+--------------------------+---------------------------+
---------------------Strategy_Recommand---------------------
Recommand_Strategy:
+--------+------+------+------+---------------------------+-----------------+
| Zero   |   DP |   TP |   PP |   Train_momery_per_gpu/GB |   Trianing_days |
+========+======+======+======+===========================+=================+
| Zero1  |   80 |    1 |    1 |                      8.44 |            0.01 |
+--------+------+------+------+---------------------------+-----------------+
Please find the best batch_size by adjusting BATCH_SIZE
```