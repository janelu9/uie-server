# UIE Inference Service
[UIE](https://gitee.com/paddlepaddle/PaddleNLP/tree/develop/applications/information_extraction/text) model's inference service for information extraction. Supportting infinite length of text inputs, faster and more accurate.
  ```python
>>> from server import UIEInferModel,static_model_file,static_params_file
>>> uie =UIEInferModel(static_model_file,static_params_file)
>>> schema=['时间', '人物', '赛事名称'] # Define the schema for entity extraction
>>> text="2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"
>>> print(uie(schema,text))
{'时间': [{'text': '2月8日上午',
   'start': 0,
   'end': 6,
   'probability': 0.9857378532473966}],
 '人物': [{'text': '谷爱凌',
   'start': 28,
   'end': 31,
   'probability': 0.9986867948418592}],
 '赛事名称': [{'text': '北京冬奥会自由式滑雪女子大跳台决赛',
   'start': 6,
   'end': 23,
   'probability': 0.8503081645734305}]}
   ```
