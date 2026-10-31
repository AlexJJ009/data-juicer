# AGENTS.md

给 Codex、Claude Code 的长期开发规则。`CLAUDE.md` 只引用本文件。服务器、启动方式、hook 和分工的规则在服务器的 `~/.codex/AGENTS.md`。

本仓库是 [Data-Juicer](https://github.com/datajuicer/data-juicer) 的 fork，在上游基础上增加科学数据的精炼算子。上游的开发规范（算子写法、依赖懒加载、代码风格）见 `docs/DeveloperGuide.md`；与本文件冲突时以本文件为准。

## 改动原则

- 能随时合并上游：新功能放在新模块里，上游文件只做注册所需的最小改动；不格式化、不重排、不批量改名上游代码。
- 按领域分目录：算子放 `data_juicer/ops/domain/<领域>/`，测试放 `tests/ops/domain/<领域>/`。
- 先用上游已有的基类、工具和成熟库，不重新实现已有的算法。第三方库用 `LazyLoader` 引入，不加进核心依赖。
- 不改上游已有算子、执行器和 tracer（tracer 不在本项目范围）。
- 不夹带无关改动，不复制粘贴大段代码。注释、测试名和提交说明用英文，与上游一致。

## 测试

- 改行为先写会失败的测试，确认它失败，再实现到通过。修 bug 先写复现测试。
- 测试和代码在同一次提交里，目录与源码对应。删除或改名功能时同步删改测试。
- 单元测试用小的合成数据，不依赖网络、GPU 和服务器上的大文件。需要真实数据、要跑很久的检查写成实验脚本，输出放 `runs/`，不放进单元测试。
- 合入 main 前必须通过：改动所在领域的测试，以及改动文件上的 pre-commit。

  ```bash
  python -m pytest tests/ops/domain/<领域> -q
  pre-commit run --files <改动的文件>
  ```

- 测试失败不靠跳过、删断言或标记 xfail 过关。上游原有的失败测试要在 REPORT 里注明，并说明与本次改动无关的依据。

## 工具

- 环境：用提示词指定的 `envs/` 下的环境；不往 `.venv` 和别人的环境装包。依赖改动写进 `pyproject.toml` 并更新 `uv.lock`。
- 格式和检查：用仓库的 pre-commit 配置（isort、black 等），只对改动文件运行。
- git：Codex 只用只读命令（`status`、`diff`、`log`、`show`）和 `worktree add`；commit、merge、rebase、reset、push、改 remote 由 Claude 做。

## 分支

- `main` = 上游 + 我们合入的改动。开发在 worktree 的功能分支上，Claude 审查后合入 main。
- remote：`origin` 是 fork，`upstream` 是官方仓库（push 已禁用）。同步上游由 Claude 用 merge 完成。
