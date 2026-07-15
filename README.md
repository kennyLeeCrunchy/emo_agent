# Emo Agent

中文语音情绪陪伴应用原型，包含 React 前端和 FastAPI 后端，支持录音或上传音频、查看分析结果，并进行带会话记忆的陪伴式对话。情绪分析仅作参考，不构成心理或医疗诊断。

## 本地运行

需要 Python 3.10+ 和 Node.js。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python -m backend.run_fastapi --host 127.0.0.1 --port 7860
```

前端另开一个终端：

```powershell
cd frontend
npm ci
npm run dev
```

完整语音分析需要自行配置本地推理权重。DeepSeek 对话功能可通过环境变量配置 `DEEPSEEK_API_KEY`；不要把密钥提交到仓库。

## 开发

```powershell
pytest
```

第三方数据集和模型的来源与权利说明见 [`DATASET_NOTICE.md`](DATASET_NOTICE.md)。
