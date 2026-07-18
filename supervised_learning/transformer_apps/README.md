## Resources

**Read or watch:**


- [How machines Read](/rltoken/kd5yrDx2tlh1LfeCWBvpOA)
- [Sub-word tokenizers](/rltoken/luuEHmn8yqarthPbrei8KA)
- [Summary of the tokenizers](/rltoken/aBx_d0p9xhcyYdo6cjP0sw)
- [Subword Tokenization](/rltoken/KgJI9YXjBOmRf9RCthqv9g)
- [Notes on BERT tokenizer and model](/rltoken/2DH5A0_Bm5rRLYvMBe6MAA)
- [What is AutoTokenizer?](/rltoken/tqHc6InG4aMGhGRGk8or8g)
- [Training a new tokenizer from an old one](/rltoken/ObRBDgX-h_ZP_bjwy63pew)
- [TFDS Overview](/rltoken/GGUi9ziJ5vLQbHqT3aPYSw)
- [How Transformers Work: A Detailed Exploration of Transformer Architecture](/rltoken/KUgtQKtHcwD8vr8cpPffnQ)

**References:**

- [tfds](/rltoken/IZRkjPgMMT38dzSYJzhWvQ)
	- [tfds.load](/rltoken/K0ilcVeOihLDheWFMZgAPQ)
* [AutoTokenizer](/rltoken/pkUfcNxQl2gmsMC_25Oz2w)
* [train_new_from_iterator](/rltoken/VpO4fhAS3q8tP4cmIiNOIA)
* [encode](/rltoken/lCC6XnySm1bdtWbwl_sBzA)
- [tf.py\_function](/rltoken/P8RV_fpQFZGKHClzEAxnQg)
- [TFDS Keras Example](/rltoken/0brH315xMXBHt-Ni8aJQhA)
- [tf.linalg.band\_part](/rltoken/qrowNd_CSj2TAuGSGUUX5A)
- [Customizing what happens in fit](/rltoken/exVwAMC08C91jNYHYp_JSg)


## Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](/rltoken/tSsTJf_8csoUSSTXyleseg), __without the help of Google__:

### General

- How to use Transformers for Machine Translation
- How to write a custom train/test loop in Keras
- How to use Tensorflow Datasets


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
- Unless otherwise stated, you cannot import any module except `import transformers` and `from setup import load_pt2en` (the dataset helper described in the **TF Datasets** section). For some tasks, `import tensorflow as tf` is also allowed.


## TF Datasets

For machine translation, we use the Portuguese-to-English subset of [ted_hrlr_translate](/rltoken/bk9w62FECGXMsqdJC_qKVw), a parallel corpus extracted from TED talks.

### Heads up: manual setup required

The original `tfds.load(&#39;ted_hrlr_translate/pt_to_en&#39;, ...)` call can no longer download the data because the upstream archive on phontron.com is offline. Holberton hosts a mirror of the dataset and a small Python helper that wraps it with the same interface as `tfds.load`. Follow the three steps below before starting the project.

### Step 1. Download and extract the dataset

```
curl -L -O https://holbucket-prod.s3.fr-par.scw.cloud/projects/2422/ted_hrlr_pt_to_en.tar.gz
mkdir -p ~/.cache/ted_hrlr
tar -xzvf ted_hrlr_pt_to_en.tar.gz -C ~/.cache/ted_hrlr
```

After extraction you should have the following parallel files under `~/.cache/ted_hrlr/datasets/pt_to_en/`:

- `pt.train` and `en.train` (training split)
- `pt.dev` and `en.dev` (validation split)
- `pt.test` and `en.test` (test split)

### Step 2. Download the `setup.py` helper

Place `setup.py` at the root of your project folder, next to your `0-dataset.py`, `1-dataset.py`, and so on:

```
curl -L -O https://holbucket-prod.s3.fr-par.scw.cloud/projects/2422/setup.py
```

The helper exposes a single function `load_pt2en(split)` that returns a `tf.data.Dataset` of `(pt, en)` `tf.string` pairs. It is a drop-in replacement for `tfds.load(&#39;ted_hrlr_translate/pt_to_en&#39;, split=split, as_supervised=True)`. Valid splits are `&#39;train&#39;`, `&#39;validation&#39;`, and `&#39;test&#39;`.

You can verify the install by running:

```
$ python setup.py
Using data dir: /home/&lt;user&gt;/.cache/ted_hrlr/datasets/pt_to_en

     train: 51785 pairs
validation: 1193 pairs
      test: 1803 pairs

First training example:
  PT: entre todas as grandes privações com que nos debatemos hoje — pensamos em financeiras e económicas primeiro — aquela que mais me preocupa é a falta de diálogo político — a nossa capacidade de abordar conflitos modernos como eles são , de ir à raiz do que eles são e perceber os agentes-chave e lidar com eles .
  EN: amongst all the troubling deficits we struggle with today — we think of financial and economic primarily — the ones that concern me most is the deficit of political dialogue — our ability to address modern conflicts as they are , to go to the source of what they &#39;re all about and to understand the key players and to deal with them .

Ready. Import load_pt2en() in your training code.
```

### Step 3. Install required Python packages

```
pip install --user tensorflow-datasets==4.9.2
```

### Loading the dataset in your code

```
$ cat load_dataset.py
#!/usr/bin/env python3
from setup import load_pt2en

pt2en_train = load_pt2en(&#39;train&#39;)
for pt, en in pt2en_train.take(1):
  print(pt.numpy().decode(&#39;utf-8&#39;))
  print(en.numpy().decode(&#39;utf-8&#39;))
$ ./load_dataset.py
entre todas as grandes privações com que nos debatemos hoje — pensamos em financeiras e económicas primeiro — aquela que mais me preocupa é a falta de diálogo político — a nossa capacidade de abordar conflitos modernos como eles são , de ir à raiz do que eles são e perceber os agentes-chave e lidar com eles .
amongst all the troubling deficits we struggle with today — we think of financial and economic primarily — the ones that concern me most is the deficit of political dialogue — our ability to address modern conflicts as they are , to go to the source of what they &#39;re all about and to understand the key players and to deal with them .
```

## Transformers

To download  transformers library, please use:

```
pip install --user transformers==4.44.2
