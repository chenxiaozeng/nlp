# 开放域信息抽取

## 目 录

* [开放域信息抽取和UIE概述](#开放域信息抽取概述)
* [应用场景和效果展示](#应用场景和效果展示)
* [一键预测](#一键预测)
* [定制训练](#定制训练)

<a name="开放域信息抽取概述"></a>
## 开放域信息抽取和UIE概述

**开放域信息抽取的介绍**......

**UIE的介绍**....Unified Information Extraction（UIE）将各种类型的信息抽取任务统一转化为自然语言的形式，并进行多任务联合训练。利用单一模型支持多种类型的开放抽取任务，用户可以使用自然语言自定义抽取目标，无需训练即可抽取输入文本中的对应信息。该模型支持多种类型的开放抽取任务，包括但不限于命名实体、关系、事件论元、事件描述片段、评价、评价维度、观点词、情感倾向等...
<details><summary>&emsp; UIE原理介绍 </summary><div>
 该算子的技术方案是，将各种类型的信息抽取任务统一转化为自然语言的形式，并进行多任务联合训练，进而支持零样本信息抽取。模型的输入是待抽取文本（content）和自然语言描述的抽取目标（prompt），prompt通常建议的结构为“A的B”或“B”的形式，如下例子：
    <img src="https://user-images.githubusercontent.com/11793384/165440075-53487b01-692d-4f2e-b4e6-7dbd26bca28d.png" align="middle">
    <img src="https://user-images.githubusercontent.com/11793384/165440110-1d40b7f8-a490-4ba8-99eb-2cf607386a84.png" align="middle">
</div></details>

<a name="应用场景和效果展示"></a>
## 应用场景和效果展示

**支持任务列表、应用场景和效果展示**
- 实体抽取
- 关系抽取
- 事件抽取
- 评论观点抽取
- 情感倾向判断

可应用于但不限于医疗、金融、政务等多个垂直领域....

<details><summary>&emsp; 应用场景和效果展示 </summary><div>

 <img src="https://user-images.githubusercontent.com/11793384/165439119-5f6a7562-9f6c-4a23-8c76-6d4177759201.png" align="middle">
    <img src="https://user-images.githubusercontent.com/11793384/165439567-b05da240-1155-49d8-a0c9-f1e11d0b9099.png" align="middle">
    <img src="https://user-images.githubusercontent.com/11793384/165439514-ceeceafc-b782-4ed7-b8bd-f4c4f901e759.png" align="middle">
    <img src="" align="middle">
</div></details>

为了方便使用，我们提供了一键预测功能，以及定制训练功能。

<a name="一键预测"></a>
## 一键预测

链接到taskflow文档...

### 输入说明

1. 自定义prompt的技巧

本算子的抽取效果和用户构造的抽取目标prompt直接相关。所以使用时建议多尝试prompt的不同说法，查看效果。一般来说建议prompt尽量和原文类似，比如(这个case不好，太规则了)：
```
>>> schema = [{"国家": ["总统", "政治家"]}]
>>> ie.set_schema(schema)
>>> ie("特朗普是美国的总统也是政治家。 ")
[{'国家': [{'text': '美国', 'start': 4, 'end': 6, 'probability': 0.9786417762034461, 'relation': {'总统': [{'text': '特朗普', 'start': 0, 'end': 3, 'probability': 0.9991580774669728}], '政治家': [{'text': '特朗普', 'start': 0, 'end': 3, 'probability': 0.9917139841174176}]}}]}]
```

2. 输入数据说明

![image](https://user-images.githubusercontent.com/11793384/165436570-57f5d3db-fbda-409c-9be7-4bf20d5b48ed.png)


3. 输入Schema说明

| 任务 | Schema说明     |
| ---- | -------- |
| 实体抽取 | ```["人物名", "组织名", "时间"]``` |
| 关系抽取 | ```[{ "电视剧": ["主演", "导演", "编剧"]}]```、```[{ "电视剧": ["主演"]}]``` |
| 事件抽取 | |
| 评论观点抽取 | |
| 情感倾向判断 | |

### 输出说明
....

<a name="定制训练"></a>
## 定制训练

### 代码结构说明

```shell
.
├── utils.py          # 数据处理工具
├── model.py          # 模型组网脚本
├── metric.py         # 模型效果验证指标脚本
├── doccano.py        # 数据标注脚本
├── train.py          # 模型训练脚本
├── evaluate.py       # 模型评估脚本
├── run_train.sh      # 模型训练命令
├── run_evaluate.sh   # 模型评估命令
└── README.md
```

### 模型输入数据格式

prompt为`实体类别标签`:

```text
{"content": "《完美和喜悦在我心中》是2003年海天出版社出版的图书，作者是阿兰科恩", "result_list": [{"text": "完美和喜悦在我心中", "start": 1, "end": 10}], "prompt": "作品名"}
{"content": "《完美和喜悦在我心中》是2003年海天出版社出版的图书，作者是阿兰科恩", "result_list": [{"text": "海天出版社", "start": 17, "end": 22}], "prompt": "机构名"}
{"content": "《完美和喜悦在我心中》是2003年海天出版社出版的图书，作者是阿兰科恩", "result_list": [{"text": "阿兰科恩", "start": 31, "end": 35}], "prompt": "人物名"}
```

prompt为`实体名称` + 的 + `关系类别标签`:

```text
{"content": "《完美和喜悦在我心中》是2003年海天出版社出版的图书，作者是阿兰科恩", "result_list": [{"text": "海天出版社", "start": 17, "end": 22}], "prompt": "完美和喜悦在我心中的出版社名称"}
{"content": "《完美和喜悦在我心中》是2003年海天出版社出版的图书，作者是阿兰科恩", "result_list": [{"text": "阿兰科恩", "start": 31, "end": 35}], "prompt": "完美和喜悦在我心中的作者"}
```

### 数据标注

我们推荐使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，本案例也打通了从标注到训练的通道，即doccano导出数据后可通过[doccano.py](./doccano.py)脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。为达到这个目的，您需要按以下标注规则在doccano平台上标注数据：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/164374314-9beea9ad-08ed-42bc-bbbc-9f68eb8a40ee.png />
    <p>图2 数据标注样例图<p/>
</div>

- 在doccano平台上，定义实体标签类别和关系标签类别，上例中需要定义的实体标签有`作品名`、`机构名`和`人物名`，关系标签有`出版社名称`和`作者`。
- 使用以上定义的标签开始标注数据，图2展示了一个标注样例。
- 当标注完成后，在 doccano 平台上导出 `jsonl` 形式的文件，并将其重命名为 `doccano.json` 后，放入 `./data` 目录下。
- 通过 [doccano.py](./doccano.py) 脚本进行数据形式转换，然后便可以开始进行相应模型训练。

```shell
python doccano.py \
    --doccano_file ./data/doccano.json \
    --save_dir ./data/ext_data \
    --negative_ratio 5
```

**备注：**
- 默认情况下 [doccano.py](./doccano.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [doccano.py](./doccano.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。

### 模型训练

下载训练好的[UIE模型](https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie/model_state.pdparams)并放入`./uie_model`中:

通过运行以下命令进行自定义UIE模型训练：

```shell
sh run_train.sh
```

### 模型评估

通过运行以下命令进行模型评估：

```shell
sh run_evaluate.sh
```

### Taskflow一键预测

通过`schema`自定义抽取目标，`task_path`指定使用标注数据训练的UIE模型。

```python
from paddlenlp import Taskflow

schema = [{"作品名": ["作者", "出版社名称"]}]

# 为任务实例设定抽取目标和定制化模型权重路径
my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best')
```
