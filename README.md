# Backdoor-medicalAI
２０２０年日本バイオインフォマティクス学会年会・第９回生命医薬情報学連合大会（IIBMP2020）ポスター発表
「肺炎診断用深層ニューラルネットワークのバックドア攻撃に対する脆弱性」

## Abstract
- 深層ニューラルネットワーク（DNN）は医療画像診断に応用されるが、外部攻撃に対する脆弱性をもつ。これまで医療診断DNNにおいてバックドア攻撃に対する脆弱性は評価されていない。本研究では、肺炎診断用DNNを題材にその調査を行う。

- Deep neural networks (DNNs) are applied to medical imaging but are vulnerable to external attacks. So far, the vulnerability of medical diagnostic DNNs to backdoor attacks has not been evaluated. In this study, we investigate the subject of a DNN for pneumonia diagnosis.

- This network makes correct predictions for clean images, but makes wrong predictions for images where triggers are present.

- visually obvious　

https://github.com/YukiM00/Backdoor-medicalAI/blob/master/chestx/smallpoison_chestx.ipynb

- IIBMP2020 Presentation

https://docs.google.com/presentation/d/e/2PACX-1vQoJAlKl0XutoxMwbCtjOCD0atPJVnyYvJa8VT-216p4S0xGclDsP6ZxLL9TXHx3oQFNw-u32Be7EQ9/pub?start=false&loop=false&delayms=5000

## check
- python 3.6
- keras 
- tensolflow 1.19
- OpenCV

## Result
- nontarget attack(normal ⇆ pneumonia)
- Error Rate[%]

|       clean data       |     backdoor data      | 
| ---------------------- | ---------------------- |
|           3.7          |          90.3          |




