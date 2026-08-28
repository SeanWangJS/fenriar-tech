# AstroPaper 专栏（Series / 专题）功能设计与实现规划

## 1. 背景与目标

### 1.1 现状与痛点
当前 AstroPaper 仅支持以下三种内容组织维度：
1. **全站时间线平铺**（`/posts`）：所有文章按发布时间倒序排列。
2. **无序标签**（`/tags`）：关键词聚合，缺乏先后递进关系。
3. **归档**（`/archives`）：纯时间维度（年份/月份）聚类。

对于硬核技术博客（如 *CUDA 编程入门 1~7 讲*、*LLM 推理加速演进*、*TensorRT 加速实战 1~5 讲*、*从零实现深度学习* 等），文章具有强烈的**篇章顺序、逻辑依赖和递进学习曲线**。平铺列表会导致读者无法按序阅读，且文章内部缺乏专栏上下文导读。

### 1.2 设计目标
* **结构化组织**：支持将多篇文章归属于同一「专栏（Series）」，并按 `1 ➔ 2 ➔ 3` 显式排序。
* **沉浸式阅读体验**：在文章页内嵌「专栏导读与目录盒」，高亮当前阅读位置。
* **独立入口**：顶部导航增加 `/series` 专栏聚合大厅与独立的专栏详情目录页。
* **专栏级上下文翻页**：文章底部的上一篇/下一篇优先按专栏篇章跳转。
* **零破坏性升级**：不影响普通独立文章，未配置 `series` 的文章表现与当前完全一致。

---

## 2. 数据结构与 Schema 设计

### 2.1 Frontmatter 扩展
在 Markdown 文章的头部 Frontmatter 中新增 `series` 相关字段：

```yaml
---
title: "CUDA 编程入门（3）：向量加法示例"
pubDatetime: 2026-08-28T10:00:00+08:00
description: "CUDA 基础向量加法实现与核函数解析"
author: "SeanWang"
tags:
  - cuda
  - gpu
# --- 新增专栏字段 ---
series:
  name: "CUDA High Performance Computing"  # 专栏唯一名称
  order: 3                                 # 专栏中的排序序号（第 3 篇）
  description: "从零开始掌握 GPU 硬件架构与高性能 CUDA Kernel 优化" # 可选：专栏简介
---
```

### 2.2 Schema 校验 (`src/content.config.ts`)
更新 Content Collection 的 Zod Schema 定义：

```ts file=src/content.config.ts
const posts = defineCollection({
  loader: glob({ pattern: "**/[^_]*.{md,mdx}", base: `./${BLOG_PATH}` }),
  schema: ({ image }) =>
    z.object({
      // ... 原有字段保持不变
      title: z.string(),
      pubDatetime: z.date(),
      description: z.string(),
      tags: z.array(z.string()).default(["others"]),
      // 新增 series 字段校验
      series: z
        .object({
          name: z.string(),
          order: z.number().int().positive(),
          description: z.string().optional(),
        })
        .optional(),
    }),
});
```

---

## 3. 页面路由与 UI 交互设计

### 3.1 页面路由规划
```text
/series                 # 专栏聚合大厅（展示所有专栏卡片、篇数、简介、更新时间）
/series/[series]        # 单个专栏目录详情页（按 order 1..N 顺序展示全集文章列表）
/posts/[slug]           # 文章详情页（内嵌专栏目录导读盒与专栏级前后翻页）
```

---

### 3.2 核心组件设计

#### 组件 1：文章页内嵌导读盒 (`src/components/SeriesNav.astro`)
展示在文章标题下方或正文开头：
```html
┌─────────────────────────────────────────────────────────────┐
│ 📌 本文收录于专栏：《CUDA High Performance Computing》（第 3 / 7 篇）  │
│ ─────────────────────────────────────────────────────────── │
│  ▸ 1. GPU 硬件架构简介                                       │
│  ▸ 2. CUDA 编程模型                                         │
│  ● 3. 向量加法示例 (当前阅读)                                 │
│  ▸ 4. 共享内存与 Bank Conflict                               │
│  ▸ 5. 并行 Reduction 优化                                    │
│  ... (可展开/折叠其余章节)                                     │
└─────────────────────────────────────────────────────────────┘
```

#### 组件 2：专栏卡片组件 (`src/components/SeriesCard.astro`)
用于 `/series` 聚合页：
* 专栏标题与图标
* 专栏描述（Description）
* 包含文章篇数（如 `共 7 篇文章`）
* 最新更新时间（基于专栏内最新文章的 `pubDatetime` / `modDatetime`）

#### 组件 3：专栏专属翻页 (`src/components/AdjacentSeriesNav.astro`)
在文章底部替换或增强现有的 `AdjacentPostNav.astro`：
* 如果文章属于专栏：
  * **上一篇** ➔ 跳转到同专栏的 `order - 1`
  * **下一篇** ➔ 跳转到同专栏的 `order + 1`
  * **专栏目录** ➔ 一键回到 `/series/[series]` 专栏首页

---

## 4. 详细开发路线图（未来实施步骤）

### 步骤 1：工具函数层 (`src/utils/getSeries.ts`)
实现专栏数据的提取与聚合逻辑：
1. `getAllSeries(posts)`: 扫描所有文章，按 `series.name` 分组，计算篇数、时间与简介。
2. `getPostsBySeries(posts, seriesName)`: 获取指定专栏的所有文章，并严格按照 `series.order` 升序排列。
3. `getAdjacentSeriesPosts(currentPost, seriesPosts)`: 获取当前文章在专栏中的前后篇。

### 步骤 2：UI 组件编写
1. 创建 `src/components/SeriesNav.astro`（文章内嵌目录导读盒）。
2. 创建 `src/components/SeriesCard.astro`（专栏卡片）。

### 步骤 3：页面路由开发
1. 创建 `src/pages/series/index.astro`（专栏列表大厅）。
2. 创建 `src/pages/series/[series].astro`（专栏详情与完整目录）。

### 步骤 4：整合文章详情页与导航栏
1. 在 `src/pages/posts/[...slug]/index.astro` 中引入 `<SeriesNav />`。
2. 在 `src/components/Header.astro` 导航栏中添加 `Series` 链接。
3. 在 `src/i18n/` 中补充多语言键值（`nav.series: "Series" / "专栏"`）。

---

## 5. 预期成果与优势
1. **打造个人技术系列电子书体验**：读者可以系统性、结构化地按顺序精读深度技术专栏。
2. **提高读者站内留存时长**：通过专栏内导读，读者看完第 1 篇后会自然点击第 2 篇，大幅增加 PV 与停留时间。
3. **极度轻量与高扩展**：纯静态 SSG 渲染，0 客户端额外 JS 开销，完美适配 Astro 架构。
