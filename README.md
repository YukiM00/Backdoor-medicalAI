# Backdoor-medicalAI
２０２０年日本バイオインフォマティクス学会年会・第９回生命医薬情報学連合大会（IIBMP2020）ポスター発表
「肺炎診断用深層ニューラルネットワークのバックドア攻撃に対する脆弱性」

## AbstractCancel changes
- 深層ニューラルネットワーク（DNN）は医療画像診断に応用されるが、外部攻撃に対する脆弱性をもつ。これまで医療診断DNNにおいてバックドア攻撃に対する脆弱性は評価されていない。本研究では、肺炎診断用DNNを題材にその調査を行う。
https://www.dropbox.com/s/pnsrku53yn1wzqy/P13IIBMP2020_abstract.pdf?dl=0

## Detail
- Backdoor attack reference
   - https://ieeexplore.ieee.org/document/8685687

- IIBMP2020 Presentation
   - https://www.dropbox.com/s/pnsrku53yn1wzqy/P13IIBMP2020_abstract.pdf?dl=0
   - https://docs.google.com/presentation/d/e/2PACX-1vQoJAlKl0XutoxMwbCtjOCD0atPJVnyYvJa8VT-216p4S0xGclDsP6ZxLL9TXHx3oQFNw-u32Be7EQ9/pub?start=false&loop=false&delayms=5000

## Requirements
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




