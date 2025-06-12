---
pipeline_tag: image-text-to-text
library_name: transformers
language:
- multilingual
tags:
- got
- vision-language
- ocr2.0
license: apache-2.0
---

<h1>General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model - HF Transformers 🤗 implementation
</h1>

[🤗 Spaces Demo](https://huggingface.co/spaces/yonigozlan/GOT-OCR-Transformers) | [🌟GitHub](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/) | [📜Paper](https://arxiv.org/abs/2409.01704)</a> 


[Haoran Wei*](https://scholar.google.com/citations?user=J4naK0MAAAAJ&hl=en), Chenglong Liu*, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu,  [Zheng Ge](https://joker316701882.github.io/), Liang Zhao, [Jianjian Sun](https://scholar.google.com/citations?user=MVZrGkYAAAAJ&hl=en), [Yuang Peng](https://scholar.google.com.hk/citations?user=J0ko04IAAAAJ&hl=zh-CN&oi=ao), Chunrui Han, [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en)



![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/6653eee7a2d7a882a805ab95/QCEFY-M_YG3Bp5fn1GQ8X.jpeg)


Tips:

GOT-OCR2 works on a wide range of tasks, including plain document OCR, scene text OCR, formatted document OCR, and even OCR for tables, charts, mathematical formulas, geometric shapes, molecular formulas and sheet music. While this implementation of the model will only output plain text, the outputs can be further processed to render the desired format, with packages like `pdftex`, `mathpix`, `matplotlib`, `tikz`, `verovio` or `pyecharts`.
The model can also be used for interactive OCR, where the user can specify the region to be recognized by providing the coordinates or the color of the region's bounding box.

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/Ucas-HaoranWei/GOT-OCR2.0).

## Usage example

### Plain text inference

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

>>> image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
>>> inputs = processor(image, return_tensors="pt").to(device)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4096,
... )

>>> processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
"R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM\nName/Phone Ext. : (...)"
```

### Plain text inference batched

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

>>> image1 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
>>> image2 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"

>>> inputs = processor([image1, image2], return_tensors="pt").to(device)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4,
... )

>>> processor.batch_decode(generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
["Reducing the number", "R&D QUALITY"]
```

### Formatted text inference

GOT-OCR2 can also generate formatted text, such as markdown or LaTeX. Here is an example of how to generate formatted text:

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

>>> image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/latex.png"
>>> inputs = processor(image, return_tensors="pt", format=True).to(device)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4096,
... )

>>> processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
"\\author{\nHanwen Jiang* \\(\\quad\\) Arjun Karpur \\({ }^{\\dagger} \\quad\\) Bingyi Cao \\({ }^{\\dagger} \\quad\\) (...)"
```

### Inference on multiple pages

Although it might be reasonable in most cases to use a “for loop” for multi-page processing, some text data with formatting across several pages make it necessary to process all pages at once. GOT introduces a multi-page OCR (without “for loop”) feature, where multiple pages can be processed by the model at once, whith the output being one continuous text.
Here is an example of how to process multiple pages at once:


```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

>>> image1 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/page1.png"
>>> image2 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/page2.png"
>>> inputs = processor([image1, image2], return_tensors="pt", multi_page=True, format=True).to(device)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4096,
... )

>>> processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
"\\title{\nGeneral OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model\n}\n\\author{\nHaoran Wei (...)"
```

### Inference on cropped patches

GOT supports a 1024×1024 input resolution, which is sufficient for most OCR tasks, such as scene OCR or processing A4-sized PDF pages. However, certain scenarios, like horizontally stitched two-page PDFs commonly found in academic papers or images with unusual aspect ratios, can lead to accuracy issues when processed as a single image. To address this, GOT can dynamically crop an image into patches, process them all at once, and merge the results for better accuracy with such inputs.
Here is an example of how to process cropped patches:

```python
>>> import torch
>>> from transformers import AutoProcessor, AutoModelForImageTextToText

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", torch_dtype=torch.bfloat16, device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

>>> image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/one_column.png"
>>> inputs = processor(image, return_tensors="pt", format=True, crop_to_patches=True, max_patches=3).to(device)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4096,
... )

>>> processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
"on developing architectural improvements to make learnable matching methods generalize.\nMotivated by the above observations, (...)"
```

### Inference on a specific region

GOT supports interactive OCR, where the user can specify the region to be recognized by providing the coordinates or the color of the region's bounding box. Here is an example of how to process a specific region:

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

>>> image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
>>> inputs = processor(image, return_tensors="pt", color="green").to(device) # or box=[x1, y1, x2, y2] for coordinates (image pixels)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4096,
... )

>>> processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
"You should keep in mind what features from the module should be used, especially \nwhen you’re planning to sell a template."
```

### Inference on general OCR data example: sheet music

Although this implementation of the model will only output plain text, the outputs can be further processed to render the desired format, with packages like `pdftex`, `mathpix`, `matplotlib`, `tikz`, `verovio` or `pyecharts`.
Here is an example of how to process sheet music:

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import verovio

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
>>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

>>> image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/sheet_music.png"
>>> inputs = processor(image, return_tensors="pt", format=True).to(device)

>>> generate_ids = model.generate(
...     **inputs,
...     do_sample=False,
...     tokenizer=processor.tokenizer,
...     stop_strings="<|im_end|>",
...     max_new_tokens=4096,
... )

>>> outputs = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
>>> tk = verovio.toolkit()
>>> tk.loadData(outputs)
>>> tk.setOptions(
...     {
...         "pageWidth": 2100,
...         "pageHeight": 800,
...         "footer": "none",
...         "barLineWidth": 0.5,
...         "beamMaxSlope": 15,
...         "staffLineWidth": 0.2,
...         "spacingStaff": 6,
...     }
... )
>>> tk.getPageCount()
>>> svg = tk.renderToSVG()
>>> svg = svg.replace('overflow="inherit"', 'overflow="visible"')
>>> with open("output.svg", "w") as f:
>>>     f.write(svg)
```
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sheet_music.svg"
alt="drawing" width="600"/>

## Citation

If you find our work helpful, please consider citing our papers 📝 and liking this project ❤️！

```bib
@article{wei2024general,
  title={General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model},
  author={Wei, Haoran and Liu, Chenglong and Chen, Jinyue and Wang, Jia and Kong, Lingyu and Xu, Yanming and Ge, Zheng and Zhao, Liang and Sun, Jianjian and Peng, Yuang and others},
  journal={arXiv preprint arXiv:2409.01704},
  year={2024}
}
@article{liu2024focus,
  title={Focus Anywhere for Fine-grained Multi-page Document Understanding},
  author={Liu, Chenglong and Wei, Haoran and Chen, Jinyue and Kong, Lingyu and Ge, Zheng and Zhu, Zining and Zhao, Liang and Sun, Jianjian and Han, Chunrui and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2405.14295},
  year={2024}
}
@article{wei2023vary,
  title={Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models},
  author={Wei, Haoran and Kong, Lingyu and Chen, Jinyue and Zhao, Liang and Ge, Zheng and Yang, Jinrong and Sun, Jianjian and Han, Chunrui and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2312.06109},
  year={2023}
}
```