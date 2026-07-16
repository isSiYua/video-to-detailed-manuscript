# Golden style example

Use this reference only for editorial style and information density. Never copy its video-specific facts into another manuscript.

## Target style

- Lead with the concrete action, claim, or conclusion. Do not warm up with presenter language.
- Keep one clear purpose per paragraph. Two to five sentences is typical, but a real list or procedure may be longer.
- Preserve exact paths, buttons, parameters, numbers, conditions, verification methods, limitations, and attributed claims.
- Combine several spoken sentences when they express one idea. State the idea once, followed by its necessary reason, example, or result.
- Do not repeat the same conclusion in an introduction, section body, and closing paragraph.
- Prefer direct verbs and short sentences. Remove phrases such as “这里主要讲的是”“可以看到”“这说明”“其价值在于” unless the speaker said them.
- Use lists only for actual enumerations or steps. Use prose for explanations.
- Attribute product claims and personal judgments: “作者认为”“作者称”“视频展示”. Do not silently turn them into verified facts.

## Example A: operational detail without transcript narration

### 使用前的两项设置

进入 `设置 → 常规 → 语言`。作者提醒，在其使用环境中，需要先解决网络访问问题才能切换中文；选择语言后还要重新打开应用，设置才会生效。

随后进入“插件”，选择 Chrome 浏览器插件并启用。视频中的用途是让工具调用现有浏览器，联网访问文献网站。

Why it works: the paragraph keeps the entry path, prerequisite, restart condition, exact plugin, and purpose, but removes “首先”“然后我们可以看到”等口语流程词。

## Example B: detailed list followed by explanation

检索完成后，结果被整理为 CSV。视频展示的字段包括：

- 作者、标题、年份、来源；
- 卷、期、页码，找不到的字段留空；
- 文献类型、DOI、被引次数；
- 是否核心、关键词、编号和来源链接。

作者随机打开来源链接进行验证。画面展示了具体论文页面，因此验证方式不是只看标题，而是实际进入文献来源页面。

Why it works: the list preserves a real enumeration; the following paragraph adds the verification method once, without restating every field。

## Example C: concise attribution and limitation

作者认为，继续在通用工具中撰写全文会消耗更多 token，而且该工具并非专门面向学术写作，因此没有把它作为全文生成工具。视频随后切换到另一平台，依次输入选题、参考文献和大纲。

Why it works: it preserves the speaker's reason and next action while clearly attributing the judgment. It does not add an editor conclusion such as “这一选择体现了工具分工的重要性”。

## Anti-pattern

Avoid this style:

> 在这一部分，作者首先向我们介绍了如何进行设置。这个设置非常重要，它可以帮助用户更好地使用该工具。设置完成之后，作者又进一步介绍了插件。由此可见，合理配置工具对于整个工作流具有重要价值。

Prefer:

> 进入 `设置 → 常规 → 语言`，选择语言后重新打开应用。随后在“插件”中启用 Chrome 浏览器插件，用于访问文献网站。

## Extended example: preserve detail without padding

The following excerpt comes from a different accepted video note. It demonstrates the target density; do not reuse its facts.

### 路径一：通过播客接触创业者、开发者和产业视角

作者是多年的播客听众，早期使用苹果播客，近几年转到小宇宙。她选择小宇宙的主要原因不是节目数量，而是小宇宙提供字幕和评论功能：听到重要内容时可以回看，也能结合其他听众的讨论理解节目。

> [!info] 画面补充
> 画面列出了作者喜欢的 7 个 AI 类播客：十字路口 Crossing、硅谷 101、What's Next｜科技早知道、OnBoard!、晚点聊 LateTalk、枫言枫语、42 章经。

#### 十字路口 Crossing：偏 AI 应用、创业与产品

《十字路口 Crossing》会邀请处在一线的 AI 创业者和开发者，让嘉宾说明自己正在做什么、观察到了什么。节目更偏 AI 应用创业和产品视角。

作者推荐《OpenAI 和 Anthropic 共同看好的 FDE：AI 时代的新岗位出现，旧分工松动｜对谈 Rolling AI》。这一期讲了如何让 AI 在企业里上岗并交付结果，也提到新岗位的机会。作者认为，如果对 AI 给职场带来的冲击感到焦虑，可以听这一期转换思维。

作者还推荐面向小白、文科生和艺术生的《对话张咋啦》。嘉宾反复强调两件事：积极行动，以及坚持用 AI 去 build、真正做出东西。另一类推荐内容更偏 Agent 实操，会解释 Agent 架构中的 memory、上下文和 `Claude.md` 等概念。

#### 硅谷 101：补充产业链、投资和全球科技视角

《硅谷 101》的特点是视角广、更新快。作者主要用它补充 AI 前沿产业链、投资以及全球科技发展的信息。她举了从浅到深讲 Harness，以及介绍热门 FDE 岗位的两期节目。

> [!info] 画面补充
> 两期节目的完整标题为：
> - `E238｜聊聊 Harness 时代 AI-First 的组织架构：从信任人到信任 AI`
> - `E240｜OpenAI 联手 PE 砸下 40 亿美元，聊聊硅谷最火新职位 FDE`

### 路径二：让 AI 新闻每天自动推送到飞书

作者认为，传统的手动搜索、逐个网站刷新闻不够高效。她让 AI 主动搜集新闻并推送到飞书，每天开始工作前直接在通讯软件里阅读整理结果。

最初，她在 Claude Code 中写了一个 Routine，让 AI 定时搜索新闻、整理内容并发送到飞书。搭建过程中持续优化两个部分：新闻来源，以及推送结果的展示形式。后来 Claude 账号被封，她把同一任务迁移到 Codex：先配置飞书，再创建一个 Automation，替代 Routine 的定时触发能力。

> [!info] 画面补充
> Automation 提示词把机器人定义为“AI 新闻播报机器人”，每天运行一次，抓取最近 24 小时的重要 AI 资讯；按板块控制条目数量；将 Hacker News 等站点作为候选来源，并围绕 AI、LLM、GPT、Claude、Gemini、OpenAI、Anthropic、machine learning 等关键词筛选。

This example is longer than Examples A–C because the source contains more distinct information. Concision comes from removing repeated wording, not from forcing every paragraph to the same short length.
