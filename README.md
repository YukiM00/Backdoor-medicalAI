# Backdoor-medicalAI
２０２０年日本バイオインフォマティクス学会年会・第９回生命医薬情報学連合大会（IIBMP2020）ポスター発表

「肺炎診断用深層ニューラルネットワークのバックドア攻撃に対する脆弱性」

## Abstract
- 深層ニューラルネットワーク（DNN）は医療画像診断に応用されるが、外部攻撃に対する脆弱性をもつ。これまで医療診断DNNにおいてバックドア攻撃に対する脆弱性は評価されていない。本研究では、肺炎診断用DNNを題材にその調査を行う。

## Detail
- Backdoor attack reference
   - https://ieeexplore.ieee.org/document/8685687

## Requirements
- python 3.6
- keras 
- tensolflow 1.19
- opencv_python

## Result
- non-targeted attack(normal ⇆ pneumonia)
- Error Rate[%]

|       clean images     |     backdoor images    | 
| ---------------------- | ---------------------- |
|           3.7          |          90.3          |




