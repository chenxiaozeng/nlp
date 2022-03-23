### NER

主readme
```python
from paddlenlp import Taskflow

# 中文分词
seg = Taskflow("word_segmentation")
seg("第十四届全运会在西安举办")
>>> ['第十四届', '全运会', '在', '西安', '举办']

# 命名实体识别
ner = Taskflow("ner", mode="accurate", entity_only=True)
ner("谷爱凌拿下北京冬奥会自由式滑雪项目的冠军")
>>> [('谷爱凌', '人物类_实体'), ('拿下', '场景事件'), ('北京冬奥会', '文化类_奖项赛事活动'), ('自由式滑雪', '事件类'), ('项目', '信息资料'), ('冠军', '人物类_概念')]

# 情感分析
senta = Taskflow("sentiment_analysis")
senta("这个产品用起来真的很流畅，我非常喜欢")
>>> [{'text': '这个产品用起来真的很流畅，我非常喜欢', 'label': 'positive', 'score': 0.9938690066337585}]

# 文本相似度
from paddlenlp import Taskflow
similarity = Taskflow("text_similarity")
similarity([["春天适合种什么花？", "春天适合种什么菜？"]])
>>> [{'text1': '春天适合种什么花？', 'text2': '春天适合种什么菜？', 'similarity': 0.8340253}]
```
更多使用方法请参考[Taskflow文档](./docs/model_zoo/taskflow.md)。


```python


......
Extract feature time to cost :0.01161503791809082 seconds
Search milvus time cost is 0.004535675048828125 seconds


Q：嘉定区南翔镇实行双门长制“门长”要求落实好哪些工作？     
A：	拦、查、问、测、记
Q：上海市黄浦区老西门街道建立的党建责任区包干机制内容是什么？    
A：	街道工作人员担任楼宇联络员，分片区对接商务楼宇所属的物业公司，引导楼宇企业共同落实严防严控任务
Q：上海市街道执行“四个统一”具体指什么？  
A：	统一由居委会干部在统一时间（每周三、五下午），递交至统一地点（社区事务受理服务中心专设窗口），街道统一收集至後台 
Q：怀柔区城管委在加强监督检查方面是如何落实的？  
A：	严格落实四方责任，保证每周2~3次深入环卫、电、气、热、公共自行车、垃圾处置等单位进行巡查，督促企业做好防疫工作，协调复工复产中存在的问题，确保安全复工复产有效落实。 
Q：华新镇“亮牌分批复工”工作方案具体内容是什么？ 
A：	所有店铺一律先贴“红牌”禁止经营，经相关部门审批後，再换贴“蓝牌”准许复工。
.....



# 开放域对话-非交互模式
>>> from paddlenlp import Taskflow
>>> dialogue = Taskflow("dialogue")
>>> dialogue(["吃饭了吗"])
['刚吃完饭,你在干什么呢?']

# 开放域对话-交互模式
>>> dialogue = Taskflow("dialogue")
>>> dialogue.interactive_mode(max_turn=3)
'''
[Human]:你好
[Bot]:你好,很高兴认识你,我想问你一下,你喜欢运动吗?
[Human]:喜欢
[Bot]:那你喜欢什么运动啊?
[Human]:篮球,你喜欢篮球吗
[Bot]:当然了,我很喜欢打篮球的
'''





# 生成式问答
from paddlenlp import Taskflow
qa = Taskflow("question_answering")
qa("中国的国土面积有多大？")
>>> [{'text': '中国的国土面积有多大？',
      'answer': '960万平方公里。'}]



# 智能写诗
from paddlenlp import Taskflow
poetry = Taskflow("poetry_generation")
poetry("林密不见人")
>>> [{'text': '林密不见人', 
      'answer': ',但闻人语响。'}]
      
      
      
      
      

'''



**
from paddlenlp import Taskflow
ddp = Taskflow("dependency_parsing")
ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 
      'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 
      'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}]


from paddlenlp import Taskflow
ddp = Taskflow("dependency_parsing")
ddp("百度是一家高科技公司")
>>> [{'word': ['百度', '是', '一家', '高科技', '公司'],
      'head': [2, 0, 5, 5, 2],
      'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]




from paddlenlp import Taskflow
corrector = Taskflow("text_correction")
corrector('遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。')
>>> [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
      'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 
      'errors': [{'position': 3, 'correction': {'竟': '境'}}]}]


from paddlenlp import Taskflow
seg = Taskflow("word_segmentation")
doc = "老舍是我国现代和当代两个文学时代的文学巨人。在中华人民共和国成立之前，他的创作以小说为主，在抗日期间 对写剧本和曲艺作品有部分创作，但在新中国成立以后就把主要精力放在了创作戏剧文学上面。他的戏剧作品当中的《茶馆》就属于代表作。《茶馆》有着独特的戏剧艺术风格，对新时期的话剧发展有些重要的促进作用。他利用旧北京裕泰茶馆作为反映社会生活变化的线索，经过一系列的人物活动衬托出旧社会的黑暗与残忍。《茶馆》有着独特的戏剧艺术风格，首先它对近代中国社会的历史变迁和时代更替展现出高超的艺术概括力。它概述了清末维新变法失败后封建顽固分子张狂一时的时代、民国初期帝国主义操纵军阀压榨社会的时代、抗日胜利后国民党腐败统治时代。每个时代他只采用了简单的事件活动就引出了各个时代的黑暗与腐败！戊戌变法失败后得意忘形的老太监取了妻子、旗人常四爷因为说了一句“大清国要完”就被捉去坐了一年的牢，甚至丢掉了铁杆庄稼；吸鸦片的烟鬼改抽白面儿，还不知羞耻地称自己福气大；到了国民党政要统管妓女，社会渣渣却当上了花花公司总经理、而一个幻想实业救国的裕泰茶馆房东秦仲义变卖自己的全部家当去建立工厂，一心做着在中国发展民族工业的美梦，到了国民党时代缺成了逆产被全部没收。这些一件件简单的事情从头到尾都反映出时代特征，形形色色的表现出政治的腐败。 再者，《茶馆》的结构方式也有着独特巧妙的特点。它的结构方式不但突破了西方传统戏剧的“三一律”的束 缚，而且又跟中国戏剧“一人一事，一线到底”的戏剧规范。他的戏剧开头点出了开始，中间缺未体到，结尾却又有出现。就像秦仲义这条线一样，他开头说了秦仲义变卖家当创立工厂实业救国，后面才提到它工厂被没收，后面还自嘲“应当劝告大家， 有钱哪，就就应该吃喝嫖赌，无作非为，可千万别干好事…”可想而知他对当时的社会有多么的失望！这种有点调侃自己的需要风格，也侧面体现出《茶馆》也具有一种幽默诙谐的风格。当中提到的国民党沈处长说“好”是不说“好”字，而是学洋人腔调说“蒿”。老舍这种讽刺的手法揭露了社会丑类的卑劣，抨击了社会的病态与荒唐！ 老舍是公认的文学需要大师，他的《茶馆》 就极富有语言特色。"
print(len(doc))
'''
89100
'''
results = seg(doc)
print(results)
'''
['老舍', '是', '我', '国', '现代', '和', '当代', '两个', '文学', '时代', '的', '文学', '巨人', '。', '在', '中华人民共和国', '成立', '之', '前', '，', '他', '的', '创作', '以', '小说', '为主', '，', '在', '抗日', '期间', '对', '写', '剧本', '和', '曲艺', '作品', '有', '部分', '创作', '，', '但', '在', '新', '中国', '成立', '以后', '就', '把', '主要', '精力', '放在', '了', '创作', '戏剧', '文学', '上面', '。', '他', '的', '戏剧', '作品', '当中', '的', '《', '茶馆', '》', '就', '属于', '代表作', '。', '《', '茶馆', '》', '有着', '独特', '的', '戏剧', '艺术', '风格', '，', '对', '新', '时期', '的', '话剧', '发展', '有些', '重要', '的', '促进', '作用', '。', '他', '利用', '旧', '北京裕泰', '茶馆', '作为', '反映', '社会生活', '变化', '的', '线索', '，', '经过', '一系列', '的', '人物', '活动', '衬托', '出', '旧', '社会', '的', '黑暗', '与', '残忍', '。', '《', '茶馆', '》', '有着', '独特', '的', '戏剧', '艺术', '风格', '，', '首先', '它', '对', '近代', '中国', '社会', '的', '历史', '变迁', '和', '时代', '更替', '展现', '出', '高超', '的', '艺术', '概括', '力', '。', '它', '概述', '了', '清末', '维新', '变法', '失败', '后', '封建', '顽固', '分子', '张狂', '一时', '的', '时代', '、', '民国初期', '帝国主义', '操纵', '军阀', '压榨', '社会', '的', '时代', '、', '抗日', '胜利', '后', '国民党', '腐败', '统治', '时代', '。', '每个', '时代', '他', '只', '采用', '了', '简单', '的', '事件', '活动', '就', '引出', '了', '各', '个', '时代', '的', '黑暗', '与', '腐败', '！', '戊戌变法', '失败', '后', '得意忘形', '的', '老太', '监', '取', '了', '妻子', '、', '旗人', '常', '四爷', '因为', '说', '了', '一句', '“', '大清国', '要', '完', '”', '就', '被捉', '去', '坐', '了', '一年', '的', '牢', '，', '甚至', '丢掉', '了', '铁杆', '庄稼', '；', '吸', '鸦片', '的', '烟鬼', '改', '抽', '白面儿', '，', '还', '不知羞耻', '地', '称', '自己', '福气', '大', '；', '到', '了', '国民党', '政要', '统管', '妓女', '，', '社会', '渣渣', '却', '当上', '了', '花花公司', '总经理', '、', '而', '一个', '幻想', '实业', '救国', '的', '裕泰', '茶馆', '房东', '秦仲义', '变卖', '自己', '的', '全部', '家', '当', '去', '建立', '工厂', '，', '一心', '做', '着', '在', '中国', '发展', '民族', '工业', '的', '美梦', '，', '到', '了', '国民党', '时代', '缺成', '了', '逆产', '被', '全', '部', '没收', '。', '这些', '一件件', '简单', '的', '事情', '从头到尾', '都', '反映', '出', '时代', '特征', '，', '形形色色', '的', '表现出', '政治', '的', '腐败', '。', ' ', '再者', '，', '《', '茶馆', '》', '的', '结构', '方式', '也有', '着', '独特', '巧妙', '的', '特点', '。', '它', '的', '结构', '方式', '不', '但', '突破', '了', '西方', '传统', '戏剧', '的', '“', '三一律', '”', '的', '束缚', '，', '而且', '又', '跟', '中国', '戏剧', '“', '一人', '一事', '，', '一线', '到底', '”', '的', '戏剧', '规范', '。', '他', '的', '戏剧', '开头', '点', '出', '了', '开始', '，', '中间', '缺', '未体', '到', '，', '结尾', '却', '又', '有', '出现', '。', '就像', '秦仲义', '这', '条', '线', '一样', '，', '他', '开头', '说', '了', '秦仲义', '变', '卖家', '当', '创立', '工厂', '实业', '救国', '，', '后面', '才', '提到', '它', '工厂', '被', '没收', '，', '后面', '还', '自嘲', '“', '应当', '劝告', '大家', '，', '有钱', '哪', '，', '就', '就', '应该', '吃喝', '嫖', '赌', '，', '无作非为', '，', '可', '千万', '别', '干', '好事', '…', '”', '可想而知', '他', '对', '当时', '的', '社会', '有', '多么', '的', '失望', '！', '这种', '有点', '调侃', '自己', '的', '需要', '风格', '，', '也', '侧面', '体现', '出', '《', '茶馆', '》', '也', '具有', '一种', '幽默', '诙谐', '的', '风格', '。', '当中', '提到', '的', '国民党', '沈处长', '说', '“', '好', '”', '是', '不', '说', '“', '好', '”', '字', '，', '而是', '学', '洋人', '腔调', '说', '“', '蒿', '”', '。', '老舍', '这种', '讽刺', '的', '手法', '揭露', '了', '社会丑类', '的', '卑劣', '，', '抨击', '了', '社会', '的', '病态', '与', '荒唐', '！', ' ', '老舍', '是', '公认', '的', '文学', '需要', '大师', '，', '他', '的', '《', '茶馆', '》', '就', '极富', '有', '语言', '特色', '。']
'''


ner = Taskflow("ner", mode="accurate", entity_only=True)  # 只返回实体/概念词
ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
ner("近日国家卫健委发布第九版新型冠状病毒肺炎诊疗方案")


from paddlenlp import Taskflow
# 默认模型
ner = Taskflow("ner", mode="accurate")
# 一键使用定制训练模型
my_ner = Taskflow("ner", mode="accurate", task_path="./custom_task_path/")


## 命名实体识别
input = "谷爱凌拿下北京冬奥会自由式滑雪决赛冠军"
>>>     工具A: [('谷爱凌', 'PER'), ('拿下', 'v'), ('北京冬奥会', 'nz'), ('自由式', 'a'), 
	       ('滑雪', 'vn'), ('决赛', 'vn'), ('冠军', 'n')]
>>> Taskflow: [('谷爱凌', '人物类_实体'), ('拿下', '场景事件'), ('北京冬奥会', '文化类_奖项赛事活动'), 
	       ('自由式滑雪', '事件类'), ('决赛', '文化类_奖项赛事活动'), ('冠军', '人物类_概念')]


input = "近日国家卫健委发布第九版新型冠状病毒肺炎诊疗方案"
>>>     工具A: [('近日', 'TIME'), ('国家卫健委', 'ORG'), ('冠状病毒肺炎', 'nz')]
>>> Taskflow: [('近日', '时间类'), ('国家卫健委', '组织机构类_国家机关'),
     	       ('发布', '场景事件'), ('第九版', '信息资料'), 
               ('新型冠状病毒肺炎', '疾病损伤类'), 
               ('诊疗', '场景事件'), ('方案', '信息资料')]

     
     
     
     
     
     
```


### 中文分词 

#### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;三种分词模式，满足各类分词需求

```python
from paddlenlp import Taskflow

# 默认模式————实体粒度分词，在精度和速度上的权衡，基于百度LAC
>>> seg = Taskflow("word_segmentation")
>>> seg("近日国家卫健委发布第九版新型冠状病毒肺炎诊疗方案")
['近日', '国家卫健委', '发布', '第九版', '新型', '冠状病毒肺炎', '诊疗', '方案']

# 快速模式————最快：实现文本快速切分，基于jieba中文分词工具
>>> seg_fast = Taskflow("word_segmentation", mode="fast")
>>> seg_fast("近日国家卫健委发布第九版新型冠状病毒肺炎诊疗方案")
['近日', '国家', '卫健委', '发布', '第九版', '新型', '冠状病毒', '肺炎', '诊疗', '方案']

# 精确模式————最准：实体粒度切分准确度最高，基于百度解语
>>> seg_accurate = Taskflow("word_segmentation", mode="accurate") 
>>> seg_accurate("近日国家卫健委发布第九版新型冠状病毒肺炎诊疗方案")
['近日', '国家卫健委', '发布', '第九版', '新型冠状病毒肺炎', '诊疗', '方案']


# 中文分词
input = "我爸是我爸，我是我爸儿"
>>>     工具A:  我  爸  是  我  爸  ，  我  是  我  爸儿
>>>     工具B:  我  爸  是  我  爸  ，  我  是  我  爸儿
>>>     工具C:  我  爸  是  我  爸  ，  我  是  我  爸儿
>>>     工具D:  我爸  是  我  爸  ，  我  是  我  爸儿
>>> Taskflow:  我爸  是  我爸  ，  我  是  我爸  儿


input = "谷爱凌拿下北京冬奥会自由式滑雪决赛冠军"
>>>     工具A:  谷  爱  凌  拿下  北京  冬奥会  自由式  滑雪  决赛  冠军
>>>     工具B:  谷爱凌  拿下  北京  冬奥会  自由式  滑雪  决赛  冠军
>>>     工具C:  谷爱  凌  拿下  北京  冬奥会  自由式  滑雪  决赛  冠军
>>>     工具D:  谷爱凌  拿下  北京  冬奥会  自由式  滑雪  决赛  冠军
>>> Taskflow:  谷爱凌  拿下  北京冬奥会  自由式滑雪  决赛  冠军




input = "你知不知道我不知道你知道我是谁"
>>>     工具A:  你  知  不  知道  我  不  知道  你  知道  我  是  谁
>>>     工具B:  你  知  不  知道  我  不  知道  你  知道  我  是  谁
>>>     工具C:  你  知  不  知道  我  不  知道  你  知道  我  是  谁
>>>     工具D:  你  知不知道  我  不  知道  你  知道  我  是  谁
>>> Taskflow:  你  知不知道  我  不知道  你  知道  我  是谁


```


 
#### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;批量样本输入，平均速度更快
  
```python
>>> from paddlenlp import Taskflow
>>> seg = Taskflow("word_segmentation")
>>> seg(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
[['第十四届', '全运会', '在', '西安', '举办'], ['三亚', '是', '一个', '美丽', '的', '城市']]
```
 
#### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;自定义词典

```python
>>> from paddlenlp import Taskflow
>>> seg = Taskflow("word_segmentation")
>>> seg("平原上的火焰宣布延期上映")
['平原', '上', '的', '火焰', '宣布', '延期', '上映']
>>> seg = Taskflow("word_segmentation", user_dict="user_dict.txt")
>>> seg("平原上的火焰计划于年末上映")
['平原上的火焰', '宣布', '延期', '上映']
```
#### 参数说明
* `mode`：指定分词模式，默认为None。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `user_dict`：自定义词典文件路径，默认为None。
* `task_path`：自定义任务路径，默认为None。








<p align="center">
 <img src="../../docs/imgs/paddlenlp.png" align="middle" width="80">
<p align="center">
  
<div align="center">  
  <h1> PaddleNLP Taskflow </h1>
</div>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="https://pypi.org/project/paddlenlp/"><img src="https://img.shields.io/pypi/v/paddlenlp.svg?label=pip&logo=PyPI&logoColor=white"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleNLP?color=ffa"></a>
    <a href="https://pypi.org/project/paddlenlp/"><img src="https://img.shields.io/pypi/pyversions/paddlenlp"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg"></a>
    <a href="../../LICENSE"><img src="https://img.shields.io/github/license/paddlepaddle/paddlenlp"></a>
</p>

<h4 align="center">
    <a href="https://github.com/hankcs/HanLP/tree/master">English</a> |
    <a href="https://hanlp.hankcs.com/docs/">文档</a> |
    <a href="https://bbs.hankcs.com/">论坛</a> |
    <a href="https://mybinder.org/v2/gh/hankcs/HanLP/doc-zh?filepath=plugins%2Fhanlp_demo%2Fhanlp_demo%2Fzh%2Ftutorial.ipynb">▶️在线运行</a>
</h4>


## 特性
PaddleNLP提供**开箱即用**的产业级NLP预置任务能力，无需训练，一键预测。
- 统一的应用范式：通过`paddlenlp.Taskflow`调用，简捷易用；
- 最全的中文任务：覆盖自然语言理解与自然语言生成两大核心应用；
- 极致的产业级效果：在多个中文场景上提供产业级的精度与预测性能。


| 任务名称  | 调用方式  | 一键预测 | 单条输入 | 多条输入 | 无限长度输入 | 定制化训练 | 其它特性 | 
| :------------  | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 中文分词 | `Taskflow("word_segmentation")` | ✅ | ✅ | ✅ | ✅ | ✅ | 多种模式；自定义词典 | 
| 词性标注 | `Taskflow("pos_tagging")` | ✅ | ✅ | ✅ | ✅ | ✅ |  | 
| 命名实体识别  | `Taskflow("ner")` | ✅ | ✅ | ✅ | ✅ | ✅ | 最全中文实体标签 | 
| 句法分析 |  `Taskflow("dependency_parsing")` | ✅ | ✅ | ✅ |  | ✅ |   | 
| 文本纠错 | `Taskflow("text_correction")` | ✅ | ✅ | ✅ | ✅ | ✅ |  | 
| 文本相似度 |  `Taskflow("text_similarity")` | ✅ | ✅ | ✅ |  |  |  | 
| 情感分析 |  `Taskflow("sentiment_analysis")` | ✅ | ✅ | ✅ |  |  |  | 
| 生成式问答 |  `Taskflow("question_answering")` | ✅ | ✅ | ✅ |  |  |  | 
| 智能写诗  |  `Taskflow("poetry_generation")` | ✅ | ✅ | ✅ |  |  |  | 
| 开放域对话 |  `Taskflow("dialogue")` | ✅ | ✅ | ✅ |  | ✅ | 具备多轮对话记忆功能 |
| 『解语』-知识标注 | `Taskflow("knowledge_mining")` | ✅ | ✅ | ✅ | ✅ | ✅ | 覆盖所有中文词汇的知识标注工具  |


## Quick Start

### 环境依赖
  - python >= 3.6
  - paddlepaddle >= 2.2.0
  - paddlenlp >= 2.2.0

### 
------------------------
<table border="1">
  <tr>
    <th>环境依赖</th>
    <th>快速开始</th>
  </tr>
  <tr>
    <td>January</td>
    <td>$100</td>
  </tr>
</table>


<table  border="0" width="800" height="800" align="left" cellpadding="0" cellspacing="50" bgcolor="#ffffff">
  <tr>
    <td width="260" valign="top" bgcolor="#f2f2f2">	    
		<table width="200" border="0" cellpadding="0" cellspacing="0" align="center" >
		  <tr>
			<td align="left" colspan="2"><b>环境依赖</b></td>
		  </tr>
			<td align="left">- paddlepaddle >= 2.2.0</td>
		  </tr>
		</table>
		</td>
            	<td width="80"></td>
            	<td width="480" valign="top">
                <table width="480" height="200" border="0" cellpadding="0" cellspacing="0">
                    <tr>
                        <td colspan="2"><b>快速开始</b></td>
                    </tr>
                    <tr>
                        <td>`pip install paddlenlp` <br/>`import paddlenlp`</td>
                    </tr>
                </table>
                <br/>
    </td>
  </tr>
</table>  
------------------------

Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
再测试下  |  可替换成图片

------------------------
 ### 环境依赖
  - python >= 3.6
  - paddlepaddle >= 2.2.0
  - paddlenlp >= 2.2.0
 
------------------------



<table>
<tr><th colspan='6'>WordTag标签集合
<tr><td>人物类_实体<td>物体类<td>生物类_动物<td>医学术语类<td>链接地址<td>肯定词
<tr><td>人物类_概念<td>物体类_兵器<td>品牌名<td>术语类_生物体<td>个性特征<td>否定词
<tr><td>作品类_实体<td>物体类_化学物质<td>场所类<td>疾病损伤类<td>感官特征<td>数量词
<tr><td>作品类_概念<td>其他角色类<td>场所类_交通场所<td>疾病损伤类_植物病虫害<td>场景事件<td>叹词
<tr><td>组织机构类<td>文化类<td>位置方位<td>宇宙类<td>介词<td>拟声词
<tr><td>组织机构类_企事业单位<td>文化类_语言文字<td>世界地区类<td>事件类<td>介词_方位介词<td>修饰词
<tr><td>组织机构类_医疗卫生机构<td>文化类_奖项赛事活动<td>饮食类<td>时间类<td>助词<td>外语单词
<tr><td>组织机构类_国家机关<td>文化类_制度政策协议<td>饮食类_菜品<td>时间类_特殊日<td>代词<td>英语单词
<tr><td>组织机构类_体育组织机构<td>文化类_姓氏与人名<td>饮食类_饮品<td>术语类<td>连词<td>汉语拼音
<tr><td>组织机构类_教育组织机构<td>生物类<td>药物类<td>术语类_符号指标类<td>副词<td>词汇用语
<tr><td>组织机构类_军事组织机构<td>生物类_植物<td>药物类_中药<td>信息资料<td>疑问词<td>w(标点)
</table>
	
---------------------------

-----------------------------------
<img align="right" height="88" src="https://www.rasa.com/assets/img/sara/sara-open-source-2.0.png" alt="An image of Sara, the Rasa mascot bird, holding a flag that reads Open Source with one wing, and a wrench in the other" title="Rasa Open Source">

<p align="right">
    <a href="https://pypi.org/project/paddlenlp/"><img src="https://img.shields.io/pypi/v/paddlenlp.svg?label=pip&logo=PyPI&logoColor=white"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleNLP?color=ffa"></a>
</p>




## 详细使用

### PART I、一键预测

<details><summary><b>中文分词</b></summary><div>

### 中文分词
  
#### 三种分词模式，满足所有分词需求

```python
from paddlenlp import Taskflow

# 默认模式————实体粒度分词，又快又准
seg = Taskflow("word_segmentation")
seg(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
>>> [['第十四届', '全运会', '在', '西安', '举办'], ['三亚', '是', '一个', '美丽', '的', '城市']]

# 快速模式————集成jieba中文分词工具，实现文本快速切分
seg = Taskflow("word_segmentation", mode="fast")
seg("次级抵押贷款危机和信用违约掉期危机最大的区别是什么？")
>>> ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',]

# 精确模式————实体粒度切分准确度最高
seg = Taskflow("word_segmentation", mode="accurate") 
seg("李伟拿出具有科学性、可操作性的《陕西省高校管理体制改革实施方案》")
>>> ['李伟', '拿出', '具有', '科学性', '、', '可操作性', '的', '《', '陕西省高校管理体制改革实施方案', '》']
```

#### 自定义词典

你可以通过传入`user_dict`参数，装载自定义词典来定制分词结果。
在默认模式和精确模式下，词典文件每一行为一个自定义item。词典文件`user_dict.txt`示例：
```text
平原上的火焰
年
末
```
在快速模式下，词典文件每一行为一个自定义item+"\t"+词频。词典文件`user_dict.txt`示例：

```text
平原上的火焰  10
年 20
末 15
```

加载自定义词典及输出结果示例：
```python
from paddlenlp import Taskflow
seg = Taskflow("word_segmentation")
seg("平原上的火焰计划于年末上映")
>>> ['平原', '上', '的', '火焰', '计划', '于', '年', '末', '上映']
seg = Taskflow("word_segmentation", user_dict="user_dict.txt")
>>> ['平原上的火焰', '计划', '于', '年', '末', '上映']
```
#### 参数说明
* `mode`：指定分词模式，默认为None。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `user_dict`：自定义词典文件路径，默认为None。
* `task_path`：自定义任务路径，默认为None。
</div></details>


<details><summary><b>词性标注</b></summary><div>
  
### 词性标注

```python
from paddlenlp import Taskflow

tag = Taskflow("pos_tagging")
tag("第十四届全运会在西安举办")
>>>[('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')]

tag(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
>>> [[('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')], [('三亚', 'LOC'), ('是', 'v'), ('一个', 'm'), ('美丽', 'a'), ('的', 'u'), ('城市', 'n')]]
```

#### 标签集合

| 标签 | 含义     | 标签 | 含义     | 标签 | 含义     | 标签 | 含义     |
| ---- | -------- | ---- | -------- | ---- | -------- | ---- | -------- |
| n    | 普通名词 | f    | 方位名词 | s    | 处所名词 | t    | 时间     |
| nr   | 人名     | ns   | 地名     | nt   | 机构名   | nw   | 作品名   |
| nz   | 其他专名 | v    | 普通动词 | vd   | 动副词   | vn   | 名动词   |
| a    | 形容词   | ad   | 副形词   | an   | 名形词   | d    | 副词     |
| m    | 数量词   | q    | 量词     | r    | 代词     | p    | 介词     |
| c    | 连词     | u    | 助词     | xc   | 其他虚词 | w    | 标点符号 |
| PER  | 人名     | LOC  | 地名     | ORG  | 机构名   | TIME | 时间     |

#### 自定义词典

你可以通过装载自定义词典来定制化分词和词性标注结果。词典文件每一行表示一个自定义item，可以由一个单词或者多个单词组成，单词后面可以添加自定义标签，格式为`item/tag`，如果不添加自定义标签，则使用模型默认标签。

词典文件`user_dict.txt`示例：

```text
赛里木湖/LAKE
高/a 山/n
海拔最高
湖 泊
```

以"赛里木湖是新疆海拔最高的高山湖泊"为例，原本的输出结果为：

```text
[('赛里木湖', 'LOC'), ('是', 'v'), ('新疆', 'LOC'), ('海拔', 'n'), ('最高', 'a'), ('的', 'u'), ('高山', 'n'), ('湖泊', 'n')]
```

装载自定义词典及输出结果示例：

```python
from paddlenlp import Taskflow

my_tag = Taskflow("pos_tagging", user_dict="user_dict.txt")
my_tag("赛里木湖是新疆海拔最高的高山湖泊")
>>> [('赛里木湖', 'LAKE'), ('是', 'v'), ('新疆', 'LOC'), ('海拔最高', 'n'), ('的', 'u'), ('高', 'a'), ('山', 'n'), ('湖', 'n'), ('泊', 'n')]
```
#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `user_dict`：用户自定义词典文件，默认为None。
* `task_path`：自定义任务路径，默认为None。
</div></details>


<details><summary><b>命名实体识别</b></summary><div>

### 命名实体识别

#### 支持两种模式
  
```python
# 精确模式（默认）
from paddlenlp import Taskflow

ner = Taskflow("ner")
ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
>>> [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]

ner(["热梅茶是一道以梅子为主要原料制作的茶饮", "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"])
>>> [[('热梅茶', '饮食类_饮品'), ('是', '肯定词'), ('一道', '数量词'), ('以', '介词'), ('梅子', '饮食类'), ('为', '肯定词'), ('主要原料', '物体类'), ('制作', '场景事件'), ('的', '助词'), ('茶饮', '饮食类_饮品')], [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]]

ner = Taskflow("ner", mode="accurate", entity_only=True)  # 只返回实体/概念词
ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
>>> [('孤女', '作品类_实体'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('小说', '作品类_概念'), ('作者', '人物类_概念'), ('余兼羽', '人物类_实体')]

# 快速模式
from paddlenlp import Taskflow
ner = Taskflow("ner", mode="fast")
ner("三亚是一个美丽的城市")
>>> [('三亚', 'LOC'), ('是', 'v'), ('一个', 'm'), ('美丽', 'a'), ('的', 'u'), ('城市', 'n')]
```

#### 实体标签说明
  
- 精确模式采用的标签集合

包含66种词性及专名类别标签，标签集合如下表：

<table>

<tr><th colspan='6'>WordTag标签集合
<tr><td>人物类_实体<td>物体类<td>生物类_动物<td>医学术语类<td>链接地址<td>肯定词
<tr><td>人物类_概念<td>物体类_兵器<td>品牌名<td>术语类_生物体<td>个性特征<td>否定词
<tr><td>作品类_实体<td>物体类_化学物质<td>场所类<td>疾病损伤类<td>感官特征<td>数量词
<tr><td>作品类_概念<td>其他角色类<td>场所类_交通场所<td>疾病损伤类_植物病虫害<td>场景事件<td>叹词
<tr><td>组织机构类<td>文化类<td>位置方位<td>宇宙类<td>介词<td>拟声词
<tr><td>组织机构类_企事业单位<td>文化类_语言文字<td>世界地区类<td>事件类<td>介词_方位介词<td>修饰词
<tr><td>组织机构类_医疗卫生机构<td>文化类_奖项赛事活动<td>饮食类<td>时间类<td>助词<td>外语单词
<tr><td>组织机构类_国家机关<td>文化类_制度政策协议<td>饮食类_菜品<td>时间类_特殊日<td>代词<td>英语单词
<tr><td>组织机构类_体育组织机构<td>文化类_姓氏与人名<td>饮食类_饮品<td>术语类<td>连词<td>汉语拼音
<tr><td>组织机构类_教育组织机构<td>生物类<td>药物类<td>术语类_符号指标类<td>副词<td>词汇用语
<tr><td>组织机构类_军事组织机构<td>生物类_植物<td>药物类_中药<td>信息资料<td>疑问词<td>w(标点)

</table>

- 快速模式采用的标签集合

| 标签 | 含义     | 标签 | 含义     | 标签 | 含义     | 标签 | 含义     |
| ---- | -------- | ---- | -------- | ---- | -------- | ---- | -------- |
| n    | 普通名词 | f    | 方位名词 | s    | 处所名词 | t    | 时间     |
| nr   | 人名     | ns   | 地名     | nt   | 机构名   | nw   | 作品名   |
| nz   | 其他专名 | v    | 普通动词 | vd   | 动副词   | vn   | 名动词   |
| a    | 形容词   | ad   | 副形词   | an   | 名形词   | d    | 副词     |
| m    | 数量词   | q    | 量词     | r    | 代词     | p    | 介词     |
| c    | 连词     | u    | 助词     | xc   | 其他虚词 | w    | 标点符号 |
| PER  | 人名     | LOC  | 地名     | ORG  | 机构名   | TIME | 时间     |
  
#### 自定义词典

你可以通过装载自定义词典来定制化命名实体识别结果。词典文件每一行表示一个自定义item，可以由一个term或者多个term组成，term后面可以添加自定义标签，格式为`item/tag`，如果不添加自定义标签，则使用模型默认标签。

词典文件`user_dict.txt`示例：

```text
长津湖/电影类_实体
收/词汇用语 尾/术语类
最 大
海外票仓
```

以"《长津湖》收尾，北美是最大海外票仓"为例，原本的输出结果为：

```text
[('《', 'w'), ('长津湖', '作品类_实体'), ('》', 'w'), ('收尾', '场景事件'), ('，', 'w'), ('北美', '世界地区类'), ('是', '肯定词'), ('最大', '修饰词'), ('海外', '场所类'), ('票仓', '词汇用语')]
```

装载自定义词典及输出结果示例：

```python
from paddlenlp import Taskflow

my_ner = Taskflow("ner", user_dict="user_dict.txt")
my_ner("《长津湖》收尾，北美是最大海外票仓")
>>> [('《', 'w'), ('长津湖', '电影类_实体'), ('》', 'w'), ('收', '词汇用语'), ('尾', '术语类'), ('，', 'w'), ('北美', '世界地区类'), ('是', '肯定词'), ('最', '修饰词'), ('大', '修饰词'), ('海外票仓', '场所类')]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `user_dict`：用户自定义词典文件，默认为None。
* `task_path`：自定义任务路径，默认为None。
</div></details>

  
<details><summary><b>依存句法分析</b></summary><div>
### 依存句法分析

未分词输入:

```python
from paddlenlp import Taskflow

ddp = Taskflow("dependency_parsing")
ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}]

ddp(["9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫", "他送了一本书"])
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}, {'word': ['他', '送', '了', '一本', '书'], 'head': [2, 0, 2, 5, 2], 'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]
```

输出概率值和词性标签:

```python
ddp = Taskflow("dependency_parsing", prob=True, use_pos=True)
ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什', '球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 7, 7, 6, 6, 7, 0, 9, 10, 7], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ATT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB'], 'postag': ['TIME', 'TIME', 'PER', 'p', 'PER', 'n', 'v', 'LOC', 'n', 'PER'], 'prob': [0.79, 0.98, 1.0, 0.49, 0.97, 0.86, 1.0, 0.85, 0.97, 0.99]}]
```

使用ddparser-ernie-1.0进行预测:

```python
ddp = Taskflow("dependency_parsing", model="ddparser-ernie-1.0")
ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}]
```

使用分词结果来输入:

```python
ddp = Taskflow("dependency_parsing")
ddp.from_segments([['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫']])
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}]
```

#### 依存关系可视化

```python
from paddlenlp import Taskflow

ddp = Taskflow("dependency_parsing", return_visual=True)
result = ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")[0]['visual']
import cv2
cv2.imwrite('test.png', result)
```

#### 依存关系说明

| Label |  关系类型  | 说明                     | 示例                           |
| :---: | :--------: | :----------------------- | :----------------------------- |
|  SBV  |  主谓关系  | 主语与谓词间的关系       | 他送了一本书(他<--送)          |
|  VOB  |  动宾关系  | 宾语与谓词间的关系       | 他送了一本书(送-->书)          |
|  POB  |  介宾关系  | 介词与宾语间的关系       | 我把书卖了（把-->书）          |
|  ADV  |  状中关系  | 状语与中心词间的关系     | 我昨天买书了（昨天<--买）      |
|  CMP  |  动补关系  | 补语与中心词间的关系     | 我都吃完了（吃-->完）          |
|  ATT  |  定中关系  | 定语与中心词间的关系     | 他送了一本书(一本<--书)        |
|   F   |  方位关系  | 方位词与中心词的关系     | 在公园里玩耍(公园-->里)        |
|  COO  |  并列关系  | 同类型词语间关系        | 叔叔阿姨(叔叔-->阿姨)          |
|  DBL  |  兼语结构  | 主谓短语做宾语的结构     | 他请我吃饭(请-->我，请-->吃饭) |
|  DOB  | 双宾语结构 | 谓语后出现两个宾语       | 他送我一本书(送-->我，送-->书) |
|  VV   |  连谓结构  | 同主语的多个谓词间关系   | 他外出吃饭(外出-->吃饭)        |
|  IC   |  子句结构  | 两个结构独立或关联的单句  | 你好，书店怎么走？(你好<--走)  |
|  MT   |  虚词成分  | 虚词与中心词间的关系     | 他送了一本书(送-->了)          |
|  HED  |  核心关系  | 指整个句子的核心         |                               |

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `model`：选择任务使用的模型，可选有`ddparser`，`ddparser-ernie-1.0`和`ddparser-ernie-gram-zh`。
* `tree`：确保输出结果是正确的依存句法树，默认为True。
* `prob`：是否输出每个弧对应的概率值，默认为False。
* `use_pos`：是否返回词性标签，默认为False。
* `use_cuda`：是否使用GPU进行切词，默认为False。
* `return_visual`：是否返回句法树的可视化结果，默认为False。
* `task_path`：自定义任务路径，默认为None。
</div></details>
  

<details><summary><b>文本相似度</b></summary><div>

### 文本相似度

```python
from paddlenlp import Taskflow

similarity = Taskflow("text_similarity")
similarity([["世界上什么东西最小", "世界上什么东西最小？"]])
>>> [{'text1': '世界上什么东西最小', 'text2': '世界上什么东西最小？', 'similarity': 0.992725}]
	
	
from paddlenlp import Taskflow
similarity = Taskflow("text_similarity")
similarity([["春天适合种什么花？", "春天适合种什么菜？"]])
>>> [{'text1': '春天适合种什么花？', 'text2': '春天适合种什么菜？', 'similarity': 0.8340253}]
	
similarity([["光眼睛大就好看吗", "眼睛好看吗？"], ["小蝌蚪找妈妈怎么样", "小蝌蚪找妈妈是谁画的"]])
>>> [{'text1': '光眼睛大就好看吗', 'text2': '眼睛好看吗？', 'similarity': 0.74502707}, {'text1': '小蝌蚪找妈妈怎么样', 'text2': '小蝌蚪找妈妈是谁画的', 'similarity': 0.8192149}]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `max_seq_len`：最大序列长度，默认为128。
* `task_path`：自定义任务路径，默认为None。
</div></details>


<details><summary><b>文本纠错</b></summary><div>

### 文本纠错

```python
	
from paddlenlp import Taskflow
corrector = Taskflow("text_correction")
corrector('遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇')
>>> [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇', 'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇', 'errors': [{'position': 3, 'correction': {'竟': '境'}}]}]
	
	
from paddlenlp import Taskflow
corrector = Taskflow("text_correction")
corrector('遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。')
>>> [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
      'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 
      'errors': [{'position': 3, 'correction': {'竟': '境'}}]}]

corrector(['遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
                '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。'])
>>> [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'errors': [{'position': 3, 'correction': {'竟': '境'}}]}, {'source': '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。', 'target': '人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。', 'errors': [{'position': 18, 'correction': {'拙': '茁'}}]}]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `task_path`：自定义任务路径，默认为None。
</div></details>

  
<details><summary><b>情感分析</b></summary><div>
  
### 情感分析

```python
from paddlenlp import Taskflow
# 默认使用bilstm模型进行预测
senta = Taskflow("sentiment_analysis") 
senta("这个产品用起来真的很流畅，我非常喜欢")
>>> [{'text': '这个产品用起来真的很流畅，我非常喜欢', 'label': 'positive', 'score': 0.9938690066337585}]

senta(["这个产品用起来真的很流畅，我非常喜欢", "作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间"])
>>> [{'text': '这个产品用起来真的很流畅，我非常喜欢', 'label': 'positive', 'score': 0.9938690066337585}, {'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间', 'label': 'positive', 'score': 0.985750675201416}]

# 使用SKEP情感分析预训练模型进行预测
senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")
senta("作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。")
>>> [{'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。', 'label': 'positive', 'score': 0.984320878982544}]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `model`：选择任务使用的模型，可选有`bilstm`和`skep_ernie_1.0_large_ch`。
* `task_path`：自定义任务路径，默认为None。
</div></details>


<details><summary><b>生成式问答</b></summary><div>

### 生成式问答

```python
from paddlenlp import Taskflow

qa = Taskflow("question_answering")
qa("中国的国土面积有多大？")
>>> [{'text': '中国的国土面积有多大？', 'answer': '960万平方公里。'}]

qa(["中国国土面积有多大？", "中国的首都在哪里？"])
>>> [{'text': '中国国土面积有多大？', 'answer': '960万平方公里。'}, {'text': '中国的首都在哪里？', 'answer': '北京。'}]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
</div></details>

<details><summary><b>智能写诗</b></summary><div>

### 智能写诗

```python
from paddlenlp import Taskflow

poetry = Taskflow("poetry_generation")
poetry("林密不见人")
>>> [{'text': '林密不见人', 'answer': ',但闻人语响。'}]

poetry(["林密不见人", "举头邀明月"])
>>> [{'text': '林密不见人', 'answer': ',但闻人语响。'}, {'text': '举头邀明月', 'answer': ',低头思故乡。'}]
```

#### 可配置参数说明
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
</div></details>

  
<details><summary><b>开放域对话</b></summary><div>
### 开放域对话

#### 非交互模式
```python
from paddlenlp import Taskflow

dialogue = Taskflow("dialogue")
dialogue(["吃饭了吗"])
>>> ['刚吃完饭,你在干什么呢?']

dialogue(["你好", "吃饭了吗"], ["你是谁？"])
>>> ['吃过了,你呢', '我是李明啊']
```

可配置参数：

* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `max_seq_len`：最大序列长度，默认为512。

#### 交互模式
```python
from paddlenlp import Taskflow

dialogue = Taskflow("dialogue")
# 输入`exit`可退出交互模式
dialogue.interactive_mode(max_turn=3)

'''
[Human]:你好
[Bot]:你好,很高兴认识你,我想问你一下,你喜欢运动吗?
[Human]:喜欢
[Bot]:那你喜欢什么运动啊?
[Human]:篮球,你喜欢篮球吗
[Bot]:当然了,我很喜欢打篮球的
'''
```

交互模式参数：
* `max_turn`：任务能记忆的对话轮数，当max_turn为1时，模型只能记住当前对话，无法获知之前的对话内容。
</div></details>



### PART Ⅱ、定制化训练/自定义任务

<details><summary><b>任务列表</b></summary><div>

#### 支持定制化训练的任务列表
Taskflow提供了定制接口来使用自己的数据对模型进行微调/训练，适配任务如下：

|任务名称|默认路径||
| :---: | :---: | :---: |
|`Taskflow("word_segmentation", mode="base")`|`$HOME/.paddlenlp/taskflow/word_segmentation/lac`|[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/lexical_analysis)|
|`Taskflow("word_segmentation", mode="accurate")`|`$HOME/.paddlenlp/taskflow/word_segmentation/wordtag`|[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm)|
|`Taskflow("ner", mode="fast")`|`$HOME/.paddlenlp/taskflow/ner/lac`|[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/lexical_analysis)|
|`Taskflow("ner", mode="accurate")`|`$HOME/.paddlenlp/taskflow/ner/wordtag`|[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm)|
|`Taskflow("text_correction", model="csc-ernie-1.0")`|`$HOME/.paddlenlp/taskflow/text_correction/csc-ernie-1.0`|[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_correction/ernie-csc)|
|`Taskflow("dependency_parsing", model="ddparser")`|`$HOME/.paddlenlp/taskflow/dependency_parsing/ddparser`|[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dependency_parsing/ddparser)|
|`Taskflow("dependency_parsing", model="ddparser-ernie-1.0")`|`$HOME/.paddlenlp/taskflow/dependency_parsing/ddparser-ernie-1.0`|[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dependency_parsing/ddparser)|
|`Taskflow("dependency_parsing", model="ddparser-ernie-gram-zh")`|`$HOME/.paddlenlp/taskflow/dependency_parsing/ddparser-ernie-gram-zh`|[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dependency_parsing/ddparser)|
|`Taskflow("sentiment_analysis", model="bilstm")`|`$HOME/.paddlenlp/taskflow/sentiment_analysis/bilstm`|暂无|
|`Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")`|`$HOME/.paddlenlp/taskflow/sentiment_analysis/skep_ernie_1.0_large_ch`|[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/sentiment_analysis/skep)|
|`Taskflow("knowledge_mining", model="wordtag")`|`$HOME/.paddlenlp/taskflow/knowledge_mining/wordtag`|[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm)|
|`Taskflow("knowledge_mining", model="nptag")`|`$HOME/.paddlenlp/taskflow/knowledge_mining/nptag`|[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/nptag)|
</div></details>

<details><summary><b>定制化训练示例</b></summary><div>
  
#### 定制化训练示例

这里我们以命名实体识别`Taskflow("ner", mode="accurate")`为例，展示如何定制自己的模型。

调用`Taskflow`接口后，程序自动将相关文件下载到`$HOME/.paddlenlp/taskflow/ner/wordtag/`，该默认路径包含以下文件:

```text
$HOME/.paddlenlp/taskflow/ner/wordtag/
├── model_state.pdparams # 默认模型参数文件
├── model_config.json # 默认模型配置文件
└── tags.txt # 默认标签文件
```

* 参考上表中对应[示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/ernie-ctm)准备数据集和标签文件`tags.txt`，执行相应训练脚本得到自己的`model_state.pdparams`和`model_config.json`。

* 根据自己数据集情况，修改标签文件`tags.txt`。
  
* 将以上文件保存到任意路径中，自定义路径下的文件需要和默认路径的文件一致:

```text
custom_task_path/
├── model_state.pdparams # 定制模型参数文件
├── model_config.json # 定制模型配置文件
└── tags.txt # 定制标签文件
```
* 通过`task_path`指定自定义路径，使用Taskflow加载自定义模型进行一键预测：

```python
from paddlenlp import Taskflow
my_ner = Taskflow("ner", mode="accurate", task_path="./custom_task_path/")
```
</div></details>

## 指标说明

<details><summary><b>详情展开</b></summary><div>
  
### 分词

精确模式 vs 快速模式
  
<img src="https://user-images.githubusercontent.com/11793384/157860683-fe73d0eb-5ebf-412b-9dc7-c394dae65a23.png"  /> 

<img src="https://user-images.githubusercontent.com/11793384/157861510-c41c3e9f-d833-4533-9c9c-9ae73c3f8f30.png"  width="305" height="240" /> 

  
### 命名实体识别
<img src="https://user-images.githubusercontent.com/11793384/157860243-a8c6cda5-a5b8-452e-8342-76bf741101c6.png"  /> 

### 依存句法分析

<img src="https://user-images.githubusercontent.com/11793384/157860376-d0f087fa-40dc-4734-b623-8e9c54b0ea6a.png"  /> 
  
### 情感分析

<img src="https://user-images.githubusercontent.com/11793384/157860452-d069c4da-537d-47bf-9cf4-014532e0401e.png"  /> 
  
### 文本纠错

<img src="https://user-images.githubusercontent.com/11793384/157860487-f9725c0a-744e-4c5e-8b2f-fc935bfce114.png"  /> 
  

</div></details>
  
## FAQ

**Q：Taskflow如何修改任务保存路径？**

**A:** Taskflow默认会将任务相关模型等文件保存到`$HOME/.paddlenlp`下，可以在任务初始化的时候通过`home_path`自定义修改保存路径。

示例：
```python
from paddlenlp import Taskflow

ner = Taskflow("ner", home_path="/workspace")
```
通过以上方式即可将ner任务相关文件保存至`/workspace`路径下。

**Q：Taskflow如何自定义任务？**

**A:** 参考具体任务中的`自定义任务`说明，用户可按照示例在特定路径配置任务所需的模型权重、字典等文件，然后通过`task_path`指定自定义任务路径以一键装载任务相关文件。自然语言生成任务暂时不支持自定义任务。
  
**Q：后续会增加更多任务支持吗？**

**A:** 根据开发者反馈，持续增加中，可通过Issue或[问卷](https://wenjuan.baidu-int.com/manage/?r=survey/pageEdit&sid=85827)反馈给我们。
