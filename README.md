# Helix — 陪伴式 AI Agent

> 单用户单机部署的陪伴式 AI Agent，基于 LangGraph 编排，具备长期记忆、情感感知、主动陪伴能力。
> 后期可扩展接入微信 Bot、视频信号输入、语音对话。

---

## 一、项目定位

Helix 是一个**主打聊天的陪伴式 AI Agent**，核心目标：

- **长期陪伴**：记得用户是谁、聊过什么、关心什么
- **情感连接**：能感知用户情绪，AI 自身也有情绪状态
- **人格化**：稳定的 AI 人设（爱莉希雅），有性格、有语气、有态度
- **主动关怀**：不只是被动回答，会主动找话题、续上未完成的事
- **单机运行**：部署在个人电脑上，所有数据本地化

### 部署形态

- **当前**：CLI 交互（`app.py`），单用户单机
- **后期扩展**：
  - 微信 Clawbot 接入（远程对话触发）
  - 桌面端视频信号输入（摄像头感知用户状态）
  - 语音输入输出（VAD + ASR + TTS）

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          交互入口层                                  │
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│   │ CLI 入口  │    │ 微信Bot  │    │ 桌面应用  │    │ 语音入口  │    │
│   │ (已实现) │    │ (后期)   │    │ (后期)   │    │ (后期)   │    │
│   └─────┬────┘    └─────┬────┘    └─────┬────┘    └─────┬────┘    │
│         └────────────────┴───────────────┴───────────────┘          │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Agent 编排层 (core/)                           │
│                                                                     │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │                    Agent 生命周期                          │    │
│   │                                                           │    │
│   │  save_user_msg → start_turn → Conversation_Graph →        │    │
│   │  stream_yield → save_ai_msg → Memory_Graph(异步) →        │    │
│   │  finish_turn                                              │    │
│   └───────────────────────────────────────────────────────────┘    │
│                                                                     │
│   Conversation Graph (对话主链路)        Memory Graph (记忆写回)    │
│   ┌──────────────────────┐              ┌──────────────────────┐   │
│   │  perception          │              │  extractor           │   │
│   │    ↓                 │              │    ↓                 │   │
│   │  memory_retriever    │              │  judge               │   │
│   │    ↓                 │              │    ↓                 │   │
│   │  context_builder     │              │  updater             │   │
│   │    ↓                 │              │    ↓                 │   │
│   │  message_builder     │              │  reflection          │   │
│   │    ↓                 │              └──────────────────────┘   │
│   │  response_generator  │                                        │
│   └──────────────────────┘                                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       服务层 (services/)                            │
│                                                                     │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐   │
│   │    LLM     │ │  Memory    │ │  Context   │ │ Memory Pipe  │   │
│   │  Client    │ │  Service   │ │  Builder   │ │ Extract/Judge│   │
│   │ (DeepSeek) │ │ (3类记忆)  │ │ (Prompt)   │ │   /Updater   │   │
│   └────────────┘ └────────────┘ └────────────┘ └──────────────┘   │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐   │
│   │ Response   │ │   Event    │ │ Embedding  │ │  Proactive   │   │
│   │  Service   │ │    Bus     │ │  Provider  │ │  Scheduler   │   │
│   │ (流式生成) │ │ (错误隔离) │ │ (DashScope)│ │  (主动关怀)  │   │
│   └────────────┘ └────────────┘ └────────────┘ └──────────────┘   │
│                                                                     │
│            ★ ServiceContainer — 单例容器 / 依赖注入 ★              │
│            所有 Service 共享同一组实例，保证数据一致性              │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    运行时层 (core/runtime/)                         │
│                                                                     │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│   │ RuntimeManager  │  │  TurnManager    │  │ EventHandler    │   │
│   │ (全局状态机)    │  │ (对话轮次管理)  │  │ (事件→状态切换) │   │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                                     │
│   状态流转: LISTENING → THINKING → SPEAKING → IDLE                 │
│   Turn 模型: active + interrupted 双字段                            │
│   打断机制: 用户输入 → INTERRUPT 事件 → 清队列 + 停 TTS            │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              语音 / 多模态层 (voice/ + avatar/) — 后期接入          │
│                                                                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│   │AudioQueue│  │TTSWorker │  │ VAD/ASR  │  │ 3D Avatar│         │
│   │ (已实现) │  │ (待接入  │  │ (后期)   │  │ (后期)   │         │
│   │          │  │ 真实引擎)│  │          │  │          │         │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  基础设施层 (infrastructure/)                       │
│                                                                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│   │  MySQL   │  │  Redis   │  │  Qdrant  │  │  Video   │         │
│   │ 长期记忆 │  │ 会话历史 │  │ 向量检索 │  │  Input   │         │
│   │ (事实源) │  │ + 缓存   │  │ (Episodic)│  │ (后期)   │         │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
│                                                                     │
│   启动健康检查: Container 初始化时 ping Redis + MySQL              │
│   降级策略: Embedding/Qdrant 失败 → Episodic 降级为最近 N 条       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心数据流

### 3.1 对话主链路（Conversation Graph）

```
用户输入
   │
   ▼
[perception] ──── LLM 识别意图 + 情绪（一次调用，输出 JSON）
   │              intent: chat / comfort / knowledge / task
   │              emotion: happy / sad / angry / neutral
   ▼
[memory_retriever] ── 检索 3 类记忆
   │                   ├── Profile: 用户画像（LRU 缓存）
   │                   ├── Relationships: 当前消息提到的人物
   │                   └── Episodic: 向量检索相关历史事件（降级最近 N 条）
   ▼
[context_builder] ── 拼接 System Prompt
   │                 ├── 运行时上下文（当前时间/星期）
   │                 ├── 人设（爱莉希雅）
   │                 └── 记忆上下文（Profile + Relationships + Episodic）
   ▼
[message_builder] ── 组装 ChatRequest（system + history + user_input）
   ▼
[response_generator] ── 流式调用 LLM
   │                      ├── 发布 LLM_START / LLM_FINISH 事件
   │                      ├── 每个 StreamChunk 放入 AudioQueue
   │                      └── StreamChunk 携带 turn_id + interruptible 标记
   ▼
流式输出给用户
```

### 3.2 记忆写回链路（Memory Graph，异步执行）

```
对话结束后（后台线程，不阻塞主流程）
   │
   ▼
[extractor] ── LLM 从 (user_input, response) 中提取记忆候选
   │            ├── profile: 用户画像补丁
   │            ├── relationship: 人物关系
   │            └── episodic: 情景事件
   ▼
[judge] ────── LLM 判断候选是否值得保存（去重 / 重要度评估）
   │
   ▼
[updater] ──── 写入 3 个 Memory Service
   │            ├── ProfileService.upsert_patch（深度合并）
   │            ├── RelationshipService.upsert（实体合并/别名归并）
   │            └── EpisodicService.add（MySQL + Qdrant 双写）
   ▼
[reflection] ─ 评估本轮对话质量，写入 Episodic metadata（待实现）
```

### 3.3 打断机制

```
用户在 AI 说话时输入新消息
   │
   ▼
发布 INTERRUPT 事件
   │
   ▼
RuntimeEventHandler.on_interrupt
   │
   ├── audio_queue.clear()      # 清空未消费的 chunk
   ├── tts_worker.stop()        # 停止当前正在播放的 TTS
   └── runtime.interrupt()      # Turn 标记 interrupted=True
   │
   ▼
TTSWorker 后续 chunk 检查 is_turn_interrupted → 丢弃
```

---

## 四、记忆系统设计

### 4.1 三类记忆

| 类型 | 存储 | 用途 | 检索方式 |
|------|------|------|----------|
| **Profile** | MySQL + LRU | 用户画像（身份/偏好/性格/习惯） | 全量读取（小，LRU 缓存） |
| **Relationship** | MySQL + LRU | 用户认识的人物 + 关系 + 别名 | 按当前消息提及的人物召回 |
| **Episodic** | MySQL + Qdrant | 情景事件（含情绪/重要度/时间） | 向量语义检索（降级最近 N 条） |

### 4.2 存储分工

```
┌─────────────┐     事实源（Source of Truth）     ┌─────────────┐
│   MySQL     │ ◄──────────────────────────────► │   Qdrant    │
│             │     双写，MySQL 失败不回滚         │             │
│  全字段存储  │     Qdrant 失败 → 降级 search     │  向量索引   │
│             │     后续可补偿同步                 │  (语义检索) │
└─────────────┘                                  └─────────────┘
       ▲                                                ▲
       │                Embedding                       │
       └────────────────────────────────────────────────┘
                        DashScope API
                        (失败降级)

┌─────────────┐     会话历史（短期）              ┌─────────────┐
│   Redis     │     滑动窗口 + 滚动摘要           │  LRU Cache  │
│             │     (MAX_HISTORY_TURNS=20)        │             │
│  conv:{uid} │                                  │  Profile    │
│             │                                  │  Relationship│
└─────────────┘                                  └─────────────┘
```

### 4.3 记忆流水线

```
对话发生
   │
   ▼
MemoryExtractor (LLM)
   │  从对话中发现"可能值得长期保存"的候选
   │  输出 MemoryCandidate 列表（type + content + metadata + importance）
   ▼
MemoryJudge (LLM)
   │  判断候选是否真正值得保存
   │  去重（与已有记忆对比）
   │  调整 importance
   ▼
MemoryUpdater
   │  Profile: upsert_patch（深度合并到现有画像）
   │  Relationship: 实体合并（别名归并到 canonical_name）
   │  Episodic: MySQL INSERT → Qdrant upsert（MySQL ID 作为 Point ID）
   ▼
(后期) Reflection
      评估对话质量，写入 Episodic metadata
```

---

## 五、目录结构

```
Helix/
├── app.py                          # CLI 入口（当前）
├── config.py                       # 全局配置（环境变量驱动）
├── docker-compose.yml              # Redis + MySQL + Qdrant
├── requirements.txt
│
├── core/                           # 编排层
│   ├── agent.py                    # Agent 入口（8 步生命周期）
│   ├── conversation_graph.py       # 对话主图
│   ├── memory_graph.py             # 记忆写回图
│   ├── state.py                    # AgentState (LangGraph TypedDict)
│   │
│   ├── nodes/                      # LangGraph Node wrapper 层
│   │   ├── perception_node.py      #   感知（意图+情绪，待实现）
│   │   ├── memory_retriever_node.py#   记忆检索
│   │   ├── context_builder_node.py #   上下文构建
│   │   ├── message_builder_node.py #   消息组装
│   │   ├── response_generator_node.py # 回复生成
│   │   ├── planner_node.py         #   规划（待实现）
│   │   ├── memory_extractor_node.py#   记忆抽取
│   │   ├── memory_judge_node.py    #   记忆判断
│   │   ├── memory_updater_node.py  #   记忆更新
│   │   └── reflection_node.py      #   反思（待实现）
│   │
│   ├── runtime/                    # 运行时状态机
│   │   ├── runtime_manager.py      #   全局状态管理
│   │   ├── turn_manager.py         #   Turn 生命周期（active+interrupted）
│   │   ├── event_handler.py        #   事件 → 状态切换
│   │   ├── runtime_state.py        #   运行时状态
│   │   ├── conversation_state.py   #   对话状态枚举
│   │   └── audio_state.py          #   音频状态枚举
│   │
│   └── session/                    # 会话管理
│       ├── conversation_manager.py #   会话历史（Redis + 滑动窗口）
│       └── summarizer.py           #   滚动摘要（待实现）
│
├── services/                       # 服务层（业务实现）
│   ├── container.py                # ★ ServiceContainer 单例容器
│   │
│   ├── llm/                        # LLM 服务
│   │   ├── deepseek_client.py      #   DeepSeek 客户端
│   │   ├── client.py               #   LLM 抽象接口
│   │   ├── chat_request.py         #   请求 DTO
│   │   ├── chat_response.py        #   响应 DTO
│   │   ├── stream_chunk.py         #   流式 chunk DTO（含 turn_id）
│   │   ├── stream_processor.py     #   流处理（分句 + interruptible 标记）
│   │   └── model.py
│   │
│   ├── memory/                     # 3 类记忆服务
│   │   ├── profile_service.py      #   用户画像（LRU + MySQL）
│   │   ├── relationship_service.py #   人物关系（实体合并/别名归并）
│   │   ├── episodic_service.py     #   情景记忆（MySQL + Qdrant + TTL 缓存）
│   │   ├── retrieved_memory.py     #   检索结果 DTO
│   │   └── memory_candidate.py     #   记忆候选 DTO（Pydantic Schema）
│   │
│   ├── memory_pipeline/            # 记忆写回流水线
│   │   ├── memory_extractor.py     #   LLM 抽取记忆候选
│   │   ├── memory_judge.py         #   LLM 判断是否保存
│   │   └── memory_updater.py       #   写入 3 个 Memory Service
│   │
│   ├── context/                    # Prompt 构建
│   │   ├── context_builder.py      #   人设 + 记忆 → system prompt
│   │   ├── message_builder.py      #   组装 ChatRequest
│   │   └── system_prompt.py        #   爱莉希雅人设
│   │
│   ├── response/                   # 回复服务
│   │   └── response_service.py     #   流式生成 + chunk 处理
│   │
│   ├── event/                      # 事件总线
│   │   ├── event_bus.py            #   发布订阅（错误隔离）
│   │   └── event.py                #   事件类型枚举
│   │
│   └── embedding/                  # Embedding 服务
│       ├── embedding_base.py       #   抽象接口
│       └── dashscope_provider.py   #   DashScope 实现
│
├── voice/                          # 语音层（后期接入真实引擎）
│   ├── queue/
│   │   ├── audio_queue.py          #   音频队列（线程安全）
│   │   └── audio_item.py           #   音频项 DTO
│   └── tts/
│       └── tts_worker.py           #   TTS Worker（当前为模拟实现）
│
├── avatar/                         # 3D 模型（后期）
│
├── infrastructure/                 # 基础设施层
│   ├── database/
│   │   └── mysql.py                #   SQLAlchemy 引擎 + scoped_session
│   ├── redis/
│   │   └── redis_client.py         #   Redis 单例（懒加载）
│   └── vector/
│       ├── vector_base.py          #   向量存储抽象接口
│       └── qdrant_store.py         #   Qdrant 实现
│
├── migrations/
│   └── init.sql                    # MySQL 建表脚本
│
├── tests/                          # 测试
├── docs/                           # 文档
└── .env                            # 环境变量
```

---

## 六、技术栈

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| **LLM** | DeepSeek (OpenAI 兼容) | 意图识别 + 对话生成 + 记忆抽取/判断 |
| **编排** | LangGraph | StateGraph 驱动，两个图（Conversation + Memory） |
| **长期记忆** | MySQL 8.0 | 事实源，3 张表（profiles / relationships / episodic） |
| **短期记忆** | Redis 7 | 会话历史 + 滑动窗口 + 滚动摘要 |
| **向量检索** | Qdrant 1.12 | Episodic 语义检索，失败降级 |
| **Embedding** | DashScope | qwen3.7-text-embedding，1024 维 |
| **ORM** | SQLAlchemy 2.0 | scoped_session 线程本地 |
| **缓存** | cachetools | LRU (Profile/Relationship) + TTL (Episodic 查询) |
| **日志** | loguru | 结构化日志 |
| **运行环境** | Python 3.11 | conda 环境 remielle |

---

## 七、快速启动

### 7.1 环境准备

```bash
# 1. 进入项目目录
cd d:\PythonProject\Helix

# 2. 激活 conda 环境
conda activate remielle

# 3. 安装依赖
pip install -r requirements.txt
```

### 7.2 启动基础设施

```bash
# 启动 Redis + MySQL + Qdrant
docker-compose up -d

# 首次启动会自动执行 migrations/init.sql 建表
```

### 7.3 配置环境变量

在 `.env` 文件中配置：

```env
# LLM
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL_NAME=deepseek-chat

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=helix123

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=helix
MYSQL_PASSWORD=helix123
MYSQL_DB=helix_db

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Embedding
DASHSCOPE_API_KEY=your_dashscope_api_key
EMBEDDING_MODEL=text-embedding-v3
```

### 7.4 启动 Agent

```bash
python app.py
```

### 7.5 跳过启动健康检查（开发调试用）

```bash
# 如果只想测试部分功能，不启动 Redis/MySQL
$env:HELIX_SKIP_HEALTH_CHECK="true"
python app.py
```

---

## 八、核心设计原则

### 8.1 容器单例

所有 Service 通过 `ServiceContainer` 单例管理，避免多实例导致的状态不一致。
> 教训：早期 AudioQueue 多实例导致 TTSWorker 收不到数据，改为容器单例后解决。

### 8.2 Node / Service 双层分离

- `core/nodes/*_node.py`：LangGraph Node wrapper，只做状态读写和编排
- `services/`：业务实现，不依赖 LangGraph

Node 文件统一加 `_node.py` 后缀，避免与同名 Service 文件混淆。

### 8.3 DTO 就近放置

数据类（DTO）放在使用它的 Service 目录下：
- `StreamChunk` 在 `services/llm/`
- `AudioItem` 在 `voice/queue/`
- `MemoryCandidate` 在 `services/memory/`
- `RetrievedMemory` 在 `services/memory/`

### 8.4 Turn 双字段模型

Turn 用 `active + interrupted` 双字段区分三种状态：
- `active=True, interrupted=False`：LLM 生成中
- `active=False, interrupted=False`：正常结束，TTS 可继续播放完
- `active=False, interrupted=True`：被打断，TTS 丢弃剩余 chunk

### 8.5 CQRS-ish 记忆架构

MySQL 是事实源，Qdrant 是检索索引：
- 写入：MySQL 先写，Qdrant 后写，Qdrant 失败不回滚 MySQL
- 读取：Qdrant 检索 → MySQL 回表
- 降级：Qdrant 不可用 → `search_recent` 最近 N 条

### 8.6 优雅降级

- Embedding / Qdrant 失败 → Episodic 降级为最近 N 条
- Redis / MySQL 失败 → 启动健康检查直接报错（必选依赖）
- Memory Graph 失败 → 不影响主流程（异步执行 + 异常捕获）
- EventBus 订阅者异常 → 不影响其他订阅者（错误隔离）

---

## 九、后期扩展路线

### Phase 1：核心功能补齐（当前阶段）

- [x] EventBus 错误隔离
- [x] Memory Graph 异步执行
- [x] Redis/MySQL 启动健康检查
- [x] Episodic 缓存按 user_id 分桶
- [ ] Summarizer 滚动摘要接入
- [ ] Perception 节点（意图 + 情绪识别）
- [ ] Planner 节点（策略分流）
- [ ] LLM 失败兜底回复

### Phase 2：情感与关系深化

- [ ] AI 自身情绪状态（type + intensity + decay）
- [ ] 用户与 AI 关系亲密度模型（level 1-5）
- [ ] 记忆重要度动态调整（强化 / 衰减）
- [ ] Reflection 节点（对话质量评估）

### Phase 3：主动陪伴

- [ ] 主动话题触发器（定时 / 事件驱动）
- [ ] 未完成话题追踪（follow_up + 触发时间）
- [ ] 沉默检测与引导
- [ ] 节日 / 生日主动祝福

### Phase 4：语音对话（后期）

- [ ] 真实 TTS 引擎接入（edge-tts / DashScope TTS）
- [ ] VAD 语音活动检测（Silero / WebRTC VAD）
- [ ] ASR 语音识别（Whisper / DashScope ASR）
- [ ] 情感语调映射（emotion → TTS 语气）
- [ ] 全链路异步化（async LangGraph）

### Phase 5：多模态扩展（后期）

- [ ] 微信 Clawbot 接入（远程触发对话）
- [ ] 桌面端视频信号输入（摄像头感知用户状态）
- [ ] 3D 模型动作 / 表情系统
- [ ] 口型同步（TTS 音频驱动）

### Phase 6：工程化

- [ ] 可观测性指标（对话量 / LLM 延迟 / 记忆命中率）
- [ ] 管理后台（查看 / 修改记忆、导出历史）
- [ ] requirements 版本锁定
- [ ] app 容器化

---

## 十、Agent 8 步生命周期

每次对话严格遵循以下流程，确保状态一致：

```
1. save_user_message      保存用户消息到 Redis
        ↓
2. start_turn             创建新 Turn（turn_id），重置 Runtime 状态
        ↓
3. conversation_graph     运行对话主图（perception → retrieve → context → message → response）
        ↓
4. stream_yield           流式 yield 回复文本给调用方
        ↓
5. save_ai_message        保存 AI 回复到 Redis
        ↓
6. memory_graph (异步)    后台线程写回记忆（extractor → judge → updater → reflection）
        ↓
7. finish_turn            结束 Turn，重置 Runtime 状态
        ↓
8. (等待下次输入)
```

> 详见 [core/agent.py](file:///d:/PythonProject/Helix/core/agent.py)
