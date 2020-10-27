# Weakly-supervised learning for text in the wild detection and recognition

Repository for [Text Recognition -- Real World Data and Where to Find Them](https://arxiv.org/abs/2007.03098) and my [thesis](https://dspace.cvut.cz/handle/10467/87751) 

The code is based on [https://github.com/clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).


### Repository content:

- [x] Pretrained models
- [x] PGT files
- [x] Link to ABC dataset
- [x] Source code for running the pretrained models

### Download links

Pretrained models: [google drive](https://drive.google.com/file/d/1RtrrmL9DxiSlUROgXNLjPjJXRQyux84k/view?usp=sharing)

Link to download the ABC dataset: [github](https://github.com/uchidalab/book-dataset)

Link to download the PGT files for the ABC dataset: [google drive](https://drive.google.com/file/d/1z0z_A7S6R8ZR3BgNuoXY_wnOOEUCLu1f/view?usp=sharing)

### ABCD PGT 

Each "name".txt file corresponds to a "name".jpg file from the ABC dataset.

Each row of the PGT file corresponds to a single PGT instance in the following format:

'x<sub>1</sub> y<sub>1</sub> x<sub>2</sub> y<sub>2</sub> x<sub>3</sub> y<sub>3</sub> x<sub>4</sub> y<sub>4</sub> transcription' 
where the '\t' symbol is used as a separator ('space' can not be used because the transcription may contain spaces).

Each (x<sub>i</sub>, y<sub>i</sub>) correspond to the x and y coordinates of the i-th point of the oriented bounding box, starting with the one closest to the bottom-left corner of the image, in clock-wise order.

'transcription' is the pgt text transcription, which may consist of multiple words.

### License

Copyright 2020, Klára Janoušková

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
