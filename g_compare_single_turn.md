# G1/G2/G3 对比实验汇总（No-Judge + Judge）

## 汇总说明

- 数据集：single_turn
- no-judge 完整跑批时间：2026-04-15T19:32:30.586682
- judge 跑批时间：2026-04-15T21:37:26.658067
- 当前主文件为合并版；原始单独结果已保留为 `g_compare_single_turn_no_judge.*` 和 `g_compare_single_turn_judge.*`。
- Judge 运行异常：日志中检测到 `266` 次 `402 Insufficient Balance`。Judge 指标只可作为部分样本参考，不能当作完整 100 题全量对比。
- 有效 Judge 样本数：G1=34，G2=0，G3=0。

## 统一总览

| 分组 | no-judge Hit@K | no-judge MRR | no-judge Avg | no-judge P95 | Judge 完整性 | Judge 准确性 | Judge 可执行性 | Judge 幻觉率 | Judge 有效样本 |
|------|----------------|--------------|--------------|--------------|---------------|-------------|---------------|-------------|----------------|
| G1 | 32.0% | 0.119 | 19339ms | 33466ms | 4.12 | 4.47 | 4.91 | 3.5% | 34 |
| G2 | 31.0% | 0.310 | 18639ms | 30546ms | N/A | N/A | N/A | N/A | 0 |
| G3 | 46.0% | 0.455 | 15335ms | 21797ms | N/A | N/A | N/A | N/A | 0 |

## 结果解读

- 检索层主结论以 no-judge 全量 100 题结果为准，因为这一版没有中途余额错误。
- 生成质量层以 Judge 样本为辅；本轮只有 G1 拿到了部分有效 Judge 分，G2/G3 的 Judge 指标不可用于横向优劣判断。
- 如果后续补足余额并重跑 Judge，最合理的做法是只替换 Judge 子结果，再保留当前 no-judge 全量基线。

## no-judge 检索结论

- 1. G3: Hit@K=46.0%, MRR=0.455, Avg=15335ms, P95=21797ms
- 2. G2: Hit@K=31.0%, MRR=0.310, Avg=18639ms, P95=30546ms
- 3. G1: Hit@K=32.0%, MRR=0.119, Avg=19339ms, P95=33466ms

## Judge 运行情况

- G1: 完整性=4.12, 准确性=4.47, 可执行性=4.91, 幻觉率=3.5%, 有效样本=34
- G2: 无有效 Judge 样本，原因是本轮调用中途出现余额不足错误。
- G3: 无有效 Judge 样本，原因是本轮调用中途出现余额不足错误。

## 文件索引

- 合并汇总：`g_compare_single_turn.md` / `g_compare_single_turn_raw.json`
- no-judge 原始结果：`g_compare_single_turn_no_judge.md` / `g_compare_single_turn_no_judge_raw.json`
- judge 原始结果：`g_compare_single_turn_judge.md` / `g_compare_single_turn_judge_raw.json`