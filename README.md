# **Fine Tuning Large language models for Custom Classification tasks**

Instead of fine tuning all or last few layers of LLM, it is found that fine tuning specific layers using techniques such as LORA and QLORA can significatly improve the model performance compared to the traditional methods all while keeping the computation demand low. 

## LORA
1. Pre-trained LLMs have an intrinsic low-dimensional structure when adapted to a new task, meaning that most of the essential information can be effectively represented or approximated in a lower-dimensional subspace. 
2. LoRA allows us to train some dense layers in a neural network indirectly by optimizing rank decomposition matrices of the dense layers’ change during adaptation instead, while keeping the pre-trained weights frozen.
3. LoRA leverages the low-dimensional structure of adapted LLMs by decomposing the weight updates into smaller, low-rank matrices, reducing the number of trainable parameters and computational complexity while still capturing most of the task-relevant information.
4. LoRA is both storage- and compute-efficient.

## QLORA

QLORA, uses a novel high-precision technique to quantize a pretrained model to 4-bit, then adds a small set of learnable Low-rank Adapter weights that are tuned by backpropagating gradients through the quantized weights.
QLORA  is based on: 
1. 4-bit NormalFloat:  NF4 is a data type specifically designed particularly in the context of quantizing(Quantile Quantization-each quantization bin has an equal number of values assigned from the input tensor) the weights of neural networks to reduce memory footprints of models significantly while attempting to maintain performance.
2. Double Quantization, optimizes memory usage by quantizing quantization constants further, reducing memory footprint from 0.5 to 0.127 bits per parameter on average for a blocksize of 64, through a two-step quantization process involving 8-bit floats and symmetric quantization.
3. Paged Optimizers-leverage NVIDIA's unified memory feature for automatic CPU-GPU memory paging, preventing out-of-memory errors during gradient checkpointing for large model finetuning on a single machine.
4. LORA

We will be using QLORA technique to finetune our LLM.
