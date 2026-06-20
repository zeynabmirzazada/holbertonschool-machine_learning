## Resources

**Read or watch:**

- [MIT 6.S191: RNN](/rltoken/W9P8_tfj0q5hKIrArP5F5g)
- [Recurrent Neural Networks (RNNs), Clearly Explained!!!](/rltoken/svm0HcbvoCK_XGFbGHmZ0A)
- [Introduction to RNNs](/rltoken/bq0ElaauzPUw4mQ6Be-Uhg)
- [Illustrated Guide to RNNs](/rltoken/UAIzWrcDQL1ywicVb-2dTw)
- [Illustrated Guide to LSTM’s and GRU’s: A step by step explanation](/rltoken/70LoxqdYY_ffvocU5VOJUA)
- [Long Short-Term Memory (LSTM), Clearly Explained](/rltoken/dKp_x8uAIvhIR0nPUXivGQ)
- [Gated Recurrent Unit Networks (GRU) Explained](/rltoken/RleCjczczCHpNrM2RY4gQg)
- [RNNs Tutorial, Parts 1](/rltoken/ZPNtc-57M8l2yMazdvsdYg)
- [RNNs Tutorial, Parts 2](/rltoken/ZO1ZSzGgoLs4K9n6j0FOjg)
- [RNNs Tutorial, Parts 3](/rltoken/Dd4ey3ruOz2godixl-Xe7w)
	-  **NOTE: There is a slight mistake in the last equation for the GRU cell. It should instead be:** `s_t = (1 - z) * s_t-1 + z * h`
- [Bidirectional RNN Indepth Intuition- Deep Learning Tutorial](/rltoken/AgtxSQb_02MIwkRvu6Manw)
- [Training RNNs - Loss and BPTT](/rltoken/7Wf-NFXlLSfcBvk2ul8a0g)
- [Recurrent Neural Networks Tutorial, Part 3 – Backpropagation Through Time and Vanishing Gradients](/rltoken/E5tvAOV-Pb1PuOxPgIw1pg)
- [Deep RNNs and Bi- RNNs](/rltoken/EYmrmS0pAZKUsR0Ium5UwA)
- [Deep Recurrent Networks](/rltoken/9fNLb7HOkUa7A96DC-JAag)
- [Training and Analyzing Deep Recurrent Neural Networks](/rltoken/QFtruXrU2Xg9_xw3mrud_A)
- [Detailed explanation on  Exploding and Vanishing Gradients](/rltoken/lNLluClg2QfJbXS3VCjeww)
- [Vanishing and Exploding Gradients Problems (read until &quot;4. Gradient Clipping&quot; including)](/rltoken/bRXX_udmaKd9BsXHgg-Jiw)

**Definitions to Skim:**

- [RNN](/rltoken/sV-QJoE4WtDoplb0P8lF3A)
- [LSTM](/rltoken/Ov-7pOdRQ3ctWEuoDhqhzQ)
- [GRU](/rltoken/TgbzzZTqL9lbRDvxlNmTEA)
- [BRNN](/rltoken/bYdOCvDsJKTRTKxzNj8G1A)

## Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](/rltoken/6t7vpchNer9b-NRkByo-LA), __without the help of Google__:

### General

- What is a RNN?
- What is a LSTM?
- What is a GRU?
- What is a BRNN?
- What is the Exploding Gradient Problem? When does it occur?
- What is the Vanishing Gradient Problem? When does it occur?
- How do LSTM &amp; GRU overcome the Vanishing Gradient Problem?

## Requirements

### General


- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 20.04 LTS using `python3` (version 3.9)
- Your files will be executed with `numpy` (version 1.25.2)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.11.1)
- All your modules should have documentation (`python3 -c &#39;print(__import__(&quot;my_module&quot;).__doc__)&#39; `)
- All your classes should have documentation (`python3 -c &#39;print(__import__(&quot;my_module&quot;).MyClass.__doc__)&#39; `)
- All your functions (inside and outside a class) should have documentation (`python3 -c &#39;print(__import__(&quot;my_module&quot;).my_function.__doc__)&#39; ` and `python3 -c &#39;print(__import__(&quot;my_module&quot;).MyClass.my_function.__doc__)&#39; `)
- Unless otherwise noted, you are not allowed to import any module except `import numpy as np`
