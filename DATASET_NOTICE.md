# CSEMOTIONS 数据集来源与版权声明

本项目的语音情绪实验使用第三方数据集 **CSEMOTIONS**。本文件只说明来源和使用边界，不主张拥有该数据集、音频、文本、情绪标签、说话人标识或相关元数据的版权或其他权利。

## 来源

- 数据集发布页：[`AIDC-AI/CSEMOTIONS`](https://huggingface.co/datasets/AIDC-AI/CSEMOTIONS)
- 关联项目：[`ATH-MaaS/Marco-Voice`](https://github.com/ATH-MaaS/Marco-Voice)
- 关联论文：[`Marco-Voice Technical Report`](https://arxiv.org/abs/2508.02038)

上游数据卡将 CSEMOTIONS 描述为普通话情感语音数据集，并在元数据中标注 `Apache-2.0`。但是，关联论文的公开页面同时注明代码和数据集仅供非商业使用。由于这两处公开说明存在使用范围上的不一致，本项目不把 CSEMOTIONS 视为可以无条件商业使用或再分发的数据；使用者应以数据提供方的最新许可、说明和适用法律为准，必要时向上游作者或权利人取得单独授权。

## 本仓库的处理方式

- 本仓库不分发 CSEMOTIONS 原始音频或 parquet 文件；`CSEMOTIONS/` 仅作为本地实验目录，并被 `.gitignore` 排除。
- 本仓库代码的公开状态不会改变第三方数据集的许可，也不会授予商业使用、再分发、公开服务或声音/身份相关用途的额外权利。
- 使用者应自行确认数据来源、录音参与者权益、文本内容和下游用途符合上游条款；如需商业使用或对外提供服务，应先取得明确授权。

## 建议引用

```bibtex
@misc{tian2025marcovoicetechnicalreport,
  title={Marco-Voice Technical Report},
  author={Fengping Tian and Chenyang Lyu and Xuanfan Ni and Haoqin Sun and Qingjuan Li and Zhiqiang Qian and Haijun Li and Longyue Wang and Zhao Xu and Weihua Luo and Kaifu Zhang},
  year={2025},
  eprint={2508.02038},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2508.02038}
}
```

如果发现数据集中的内容涉及版权、隐私或其他权利问题，应联系上游数据集维护者处理；本仓库仅负责说明自身项目对该数据集的使用边界。
