-- ============================================================
-- Helix 长期记忆 MySQL 初始化脚本
-- 首次执行：mysql -uhelix -phelix123 helix_db < migrations/init.sql
-- 或挂载到 docker-entrypoint-initdb.d 随容器首次启动自动执行
-- ============================================================

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ---------------------------
-- 1. 用户画像 + 偏好（合并）
-- ---------------------------
CREATE TABLE IF NOT EXISTS profiles (
    id            BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id       VARCHAR(64)   NOT NULL COMMENT '用户/会话标识',
    profile_json  JSON          NOT NULL COMMENT '用户画像：identity/background/personality/preferences',
    created_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户画像(含偏好/习惯/厌恶)';

-- ---------------------------
-- 2. 人物/实体 + 关系
-- ---------------------------
CREATE TABLE IF NOT EXISTS relationships (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY COMMENT 'DB主键',
    user_id         VARCHAR(64)    NOT NULL COMMENT '所属用户',
    person_id       VARCHAR(64)    NOT NULL COMMENT '业务ID，如 person_001',
    canonical_name  VARCHAR(100)   NOT NULL COMMENT '人物标准名称',
    aliases_json    JSON           NOT NULL DEFAULT (JSON_ARRAY()) COMMENT '别名列表',
    relation        VARCHAR(50)    NOT NULL DEFAULT 'unknown' COMMENT '用户与此人的关系',
    extra_json      JSON           NOT NULL DEFAULT (JSON_OBJECT()) COMMENT '其他属性',
    confidence      DECIMAL(4,3)   NOT NULL DEFAULT 1.000 COMMENT '置信度 0~1',
    status          TINYINT        NOT NULL DEFAULT 1 COMMENT '1=正常 2=已合并 3=废弃',
    merged_into     VARCHAR(64)    NULL COMMENT '合并目标 person_id',
    created_at      DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_user_person (user_id, person_id),
    KEY idx_user_canonical (user_id, canonical_name),
    KEY idx_user_status (user_id, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='人物实体索引+用户关系+实体合并';

-- ---------------------------
-- 3. 情景事件记忆（含情绪 metadata + 重要度）
-- ---------------------------
CREATE TABLE IF NOT EXISTS episodic (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY COMMENT '主键=Qdrant Point ID',
    user_id         VARCHAR(64)    NOT NULL COMMENT '所属用户',
    content         TEXT           NOT NULL COMMENT '事件文本',
    tags_json       JSON           NOT NULL DEFAULT (JSON_ARRAY()) COMMENT '标签数组',
    person_ids_json JSON           NOT NULL DEFAULT (JSON_ARRAY()) COMMENT '涉及 person_id 数组',
    metadata_json   JSON           NOT NULL DEFAULT (JSON_OBJECT()) COMMENT '扩展：emotion/source/context等',
    timestamp       DATETIME       NOT NULL COMMENT '事件发生时间',
    created_at      DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '写入系统时间',
    importance      FLOAT          NOT NULL DEFAULT 0.5 COMMENT '重要度 0.0~1.0',
    KEY idx_user_time (user_id, timestamp DESC),
    KEY idx_user_importance (user_id, importance DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='情景记忆(历史事件+情绪+向量索引)';

SET FOREIGN_KEY_CHECKS = 1;
