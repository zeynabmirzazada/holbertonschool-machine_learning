## Resources:

**Read or watch:**

- [Attention Model Intuition](/rltoken/TUeicUqCo6ry17yqbo84nQ)
- [Attention Model](/rltoken/m7l8p3gQMgZn8ZS8YZhS2w)
- [How Transformers work in deep learning and NLP: an intuitive introduction](/rltoken/F7opD9DV6Jx179o2wCPfVg)
- [Transformers](/rltoken/1FuqwbxVVjBkemjRFm8rOA)
- [Bert, GPT : The Illustrated GPT-2 - Visualizing Transformer Language Models](/rltoken/FDjLspKduFkvQJur5_g3NQ)
- [SQuAD](/rltoken/0_2lRutI4Oh5-h_UviGWuQ)
- [Glue](/rltoken/qzHh-XEU-rNmcHKexAv1wQ)
- [Self supervised learning](/rltoken/4MbsOwpKehfesbS_xVIRCA)

&lt;!--
- [How Does Attention Work in Encoder-Decoder Recurrent Neural Networks](/rltoken/L2axD0KoqBeRszjix7Ucsg)
- [Attention Model](/rltoken/m7l8p3gQMgZn8ZS8YZhS2w)
- [What is a Transformer?](/rltoken/8rOQoxDFsfStrptk6HR60Q)
- [How Transformers Work](/rltoken/vZmY8FX-DRofgJVbVFK-zg)
- [Transformer: A Novel Neural Network Architecture for Language Understanding](/rltoken/6qQ30QaOTrzTSFY4EyGVBQ)
- [Stanford CS224N: NLP with Deep Learning | Winter 2019 | Lecture 14 – Transformers and Self-Attention](/rltoken/IIepHXsOsaxj-nkrTAFuPQ)
- [(Transformer) Attention Is All You Need | AISC Foundational](/rltoken/O8-fL-BsymfduxaUjg4umw)
- [Transformer Models in NLP](/rltoken/Xd8CWMJkKbdiZWD5H9qM0w)
- [Transformer model for language understanding](/rltoken/Sj9p725aJ7nTyxnbW2N6Rg)
- [Generative Modeling with Sparse Transformers](/rltoken/cdPh7uU4J1D1GAr86LncEg)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](/rltoken/aJgECzMwpc6Ew8zzlhDekg)
- [(BERT) Pretranied Deep Bidirectional Transformers for Language Understanding (algorithm) | TDLS](/rltoken/byWpLKnouMqa6haDIV4A1Q)

**References:**

- [Sequence to Sequence Learning with Neural Networks (2014)](/rltoken/dC4ZJR_96waDHffnvSznYw)
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation (2014)](/rltoken/LWSSpF1l3FNcbtHVOwo9lw)
- [Neural Machine Translation by Jointly Learning to Align and Translate](/rltoken/qBpbwkz4N2tW5_ojs0QHNg)
- [Attention Is All You Need (2017)](/rltoken/SSOB1hsM5pyEqYms1WH6UQ)
- [tf.keras.layers.Embedding](/rltoken/e5rToTsxEOhRnNjsZgfaTg)
- [tf.keras.layers.LayerNormalization](/rltoken/h9ybA7udvHKg0BAhws0DlA)
- [Improving Language Understanding by Generative Pre-Training (2018)](/rltoken/AM0rYzdGPpb8wz7RP1wKxg)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](/rltoken/Wxn2Iy4AbjpQ0Edo9Mi0HQ)
- [SQuAD 2.0](/rltoken/0_2lRutI4Oh5-h_UviGWuQ)
- [Know What You Don’t Know: Unanswerable Questions for SQuAD (2018)](/rltoken/i5oGsfzqsM8Ju-koQwz2wg)
- [GLUE Benchmark](/rltoken/PFef949u4tM4DB-GbRJCUQ)
- [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding (2019)](/rltoken/sruPWRoStl7NtMmyIjrF7w)

**More recent papers in NLP:**

- [Generating Long Sequences with Sparse Transformers (2019)](/rltoken/a6W3gzarNoY9xQcoBwl0Ww)
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (2019)](/rltoken/ECK3VKnaiRWHWISKSSc0Kg)
- [XLNet: Generalized Autoregressive Pretraining for Language Understanding (2019)](/rltoken/qV-_LLizRHYCjpYzEp4YmA)
- [Language Models are Unsupervised Multitask Learners (GPT-2, 2019)](/rltoken/NMzr_i96-3mZbUzahXDaOg)
- [Language Models are Few-Shot Learners (GPT-3, 2020)](/rltoken/RkIIzmsvHo85wCRd27QqJw)
- [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations (2020)](/rltoken/adry0GlKrOUu2p3bwC_AiQ)

To keep up with the newest papers and their code bases go to [paperswithcode.com](/rltoken/NyLw7GBAnq1bO928r1J-oA). For example, check out the [raked list of state of the art models for Language Modelling on Penn Treebank](/rltoken/t6NrBRIZeYFblEqtOcrhnQ).
--&gt;
## Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](/rltoken/kR284vgOAH-KujtFur8DaA), __without the help of Google__:

### General

- What is the attention mechanism?
- How to apply attention to RNNs
- What is a transformer?
- How to create an encoder-decoder transformer model
- What is GPT? 
- What is BERT?
- What is self-supervised learning?
- How to use BERT for specific NLP tasks
- What is SQuAD? GLUE?

## Requirements

### General

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 20.04 LTS using `python3` (version 3.9)
- Your files will be executed with `numpy` (version 1.25.2) and `tensorflow` (version 2.15)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- All of your files must be executable
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should follow the `pycodestyle` style (version 2.11.1)
- All your modules should have documentation (`python3 -c &#39;print(__import__(&quot;my_module&quot;).__doc__)&#39; `)
- All your classes should have documentation (`python3 -c &#39;print(__import__(&quot;my_module&quot;).MyClass.__doc__)&#39; `)
- All your functions (inside and outside a class) should have documentation (`python3 -c &#39;print(__import__(&quot;my_module&quot;).my_function.__doc__)&#39; ` and `python3 -c &#39;print(__import__(&quot;my_module&quot;).MyClass.my_function.__doc__)&#39; `)
- Unless otherwise stated, you cannot import any module except `import tensorflow as tf`
