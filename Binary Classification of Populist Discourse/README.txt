The code can be run all together with no issues given the right GPU and database path, reproducing plots with almost no variation in graph noises and final values.

The final, original code was run in Google Colab using a high-ram A100 GPU, which could withstand a highest batch size of 128 for BERT-tiny and of 20 for the larger models.

Results with local GPU can be obtained if the batch size is reduced to 8 for single-model runs and to 2 to 4 for running the whole code in a powerful non-server machine, naturally depending on the memory of the hardware.
As a disclaimer, of course the optimization process with low batch size will make the curves much more noisy; this means that with gradient clipping and learning rate decay the low-batch rerun results may well vary with respect to those of the project.