# Kai0 RTC 双臂评测：机械臂「抽一下 / 一抖一抖」排查与修复记录

> 日期：2026-10-26  
> 相关脚本：`scripts/eval_arx5_kai0_bimanual.py`  
> RTC 核心：`utils/kai0_rtc.py`  
> 运行日志（每次覆盖）：`data_local/kai0_eval/last_run.log`

本文记录一次真实机器人上 RTC 抖动的完整排查过程与最终改动，方便下次同类问题快速对照。

---

## 1. 现象

- 开 `--rtc` 后，双臂偶发或周期性「抽一下」「一抖一抖」。
- 不开 RTC（整 chunk 开环）时相对更顺。
- 体感节奏大约 **1.6s 一次**（与 `s_min=32 @ 20Hz` 的换 chunk 周期一致）。

---

## 2. 关键概念（先读这个）

| 名词 | 含义 |
|------|------|
| chunk / swap | 策略一次推出一段绝对关节轨迹 `[T,14]`；执行到一定步数后换成下一段 |
| `s_min` | 换 chunk 前至少执行的步数；越大换得越少 |
| `delay` / `trim` | 推理延迟对应的步数（本机约 8，对齐到 `delay_align≤8`） |
| soft | 服务器 soft-guidance 窗口长度 ≈ `exec_h - delay`；把新轨迹往旧 leftover 上拉 |
| `jump_raw0` | 换 chunk 时：**策略原始**第 0 帧相对上一拍命令的关节跳变（策略本身不连续） |
| `jump_stitch0` | stitch 后第一帧相对上一拍的跳变 |
| C0 | 位置连续（`q` 接上） |
| C1 | 速度连续（`Δq` 也接上） |
| blend | 客户端把新 chunk 前几步与旧姿态/速度做淡入 |
| leftover | 当前 chunk 尚未执行的后缀，作为 `prev_action_chunk` 送给服务器 RTC |

**重要误区**：`s_min↑` **不会**加长 soft。  
在 trim 后 `H≈42` 时：

- soft ≈ `H - switch_s` ≈ `H - s_min`
- `s_min=32` → soft≈10；`s_min=40` → soft≈8（更短）
- `s_min` 变大的好处是 **少换 chunk**（少碰边界），代价是 soft 更短

---

## 3. 根因分层（由外到内）

排查时用日志把「抖」拆成几层，不要一上来就调 `max_gw`。

### 3.1 换 chunk 边界顿挫（本次主因，已修）

**模式**：旧 chunk 末尾还在快走 → 新 chunk 第 0 步被钉成上一拍姿态 → 命令出现一步 `dq≈0` → 再加速追新轨迹。

日志特征（修复前）：

```text
hi → 0 → hi     # 例如 0.07 → 0.00 → 0.07，周期 ~1.57s
jump_stitch0 ≈ 0
step1 很小（像刹停后再起步）
```

原因有两层：

1. **只做 C0**：`actions[0] = last`，速度不连续。  
2. **把重复的第 0 步真的下发给机器人**：等于再发一次当前姿态 → 必然出现一步停住。

### 3.2 策略 chunk 间跳变大（仍存在，被边界问题放大）

```text
jump_raw0 经常 0.3~0.8（关节 max |Δ|）
```

服务器 soft + `max_gw` 压不干净时，client blend 后半段会「追赶」，体感像抽一下。  
边界 hitch 修好后，这类残留会轻很多，但不会为零。

### 3.3 速度 guard 饱和（次要）

`max_joint_step = max_joint_speed / frequency`。  
顶满 limit 时轨迹被切成极限步进，也可能发「硬」。  
本次修复后 `guard_sat` 可降到很低（例如 ~2%）。

### 3.4 其它曾踩过的坑（历史）

- leftover **零填充** → 服务器往 0 拉 → 猛抽（已改为 hold-last）。  
- 过早 swap（没等到 `commit_end`）→ 死锁/跳变。  
- `steps_per_inference` 太大一次排完整段 → leftover 空、第一次 swap 很抖。  
- 用 `(H+d)/2` 强行缩短执行段 → 换 chunk 过频，比开环更抖。

---

## 4. 本次具体改了什么

### 4.1 客户端 stitch：C1 + 丢掉重复第 0 步（关键修复）

文件：`utils/kai0_rtc.py` → `stitch_chunk_to_state`

1. 传入 `prev_velocity = last_commanded - prev_commanded`。  
2. 在 blend 窗口内沿旧速度 **coast**，再用 smoothstep 淡入新 chunk（C1）。  
3. **删除** `actions[0] == last` 的重复点，从下一拍（约 `last + v`）开始调度。

对应评测脚本：

- `scripts/eval_arx5_kai0_bimanual.py` 维护 `prev_commanded` / `last_commanded`
- 启动日志应看到：`stitch=C1+skip0`

修复后日志应满足：

- `classic hitch (hi→0→hi) = 0`
- `swap` 后第一步 `dq≈0` 的次数 = 0
- `jump_stitch0` 不再是 0，而接近换点前速度（第一拍真实要走的步长）

### 4.2 加长 client blend

```python
RTC_BLEND_STEPS = 24   # 原 16
```

换 chunk 后前半段更柔，减轻「停住再猛追」。

### 4.3 用 hold-last 垫高 `exec_h`，加长服务器 soft

文件：`utils/kai0_rtc.py` → `JointRTCInferenceWorker.maybe_start`

- 新增 `soft_min`（评测默认 `RTC_SOFT_MIN = 20`）
- `exec_h = max(real_leftover, delay + soft_min)`，多出来的部分是 leftover 的 **hold-last 填充**
- 日志里常见：`s=28 soft=20`（以前 soft≈10）

这样 **不靠盲目加大 `s_min`** 也能加长 soft。

### 4.4 推荐默认（本次跑顺时）

| 参数 | 建议 | 说明 |
|------|------|------|
| `--frequency` | 20 | 与数据采集一致 |
| `--steps-per-inference` | 8 | 且 ≤ `delay_align`，小批量调度 |
| `--rtc-s-min` | 32 | 少换与 soft 的折中；再加大 soft 会更短 |
| `--rtc-max-guidance-weight` | 2.0 | soft 引导强度；过大也可能别扭 |
| `RTC_BLEND_STEPS` | 24 | 代码常量 |
| `RTC_SOFT_MIN` | 20 | 代码常量 |
| stitch | C1+skip0 | 必须保留 |

示例：

```bash
python scripts/eval_arx5_kai0_bimanual.py \
  --policy-ip 127.0.0.1 --policy-port 8000 \
  --robot-ports 8765,8766 --camera-reorder 0,1 --no-preview \
  --rtc --frequency 20 --steps-per-inference 8
```

策略服务端需带 `--rtc`。

---

## 5. 下次再抖时：怎么快速查

### 5.1 看启动行是否带上修复

```text
[RTC] initial chunk ... blend=24 soft_min=20 stitch=C1+skip0
```

若没有 `C1+skip0`，说明跑的是旧代码。

### 5.2 看 swap 诊断

每次 swap 会打：

```text
[RTC] swap #N: cursor=... trim=... s=... soft=... H=... dt_swap=...
[RTC]   jump_raw0=... jump_stitch0=... step1=... (limit=...)
```

| 指标 | 健康大致范围 | 异常含义 |
|------|----------------|----------|
| `dt_swap` | ~1.6s（s_min=32@20Hz） | 换得过频/过慢 |
| `soft` | ≥20（有 soft_min 时） | 又变回 ~10 → soft 垫高失效 |
| `jump_raw0` | 常有 0.2~0.5 | 策略不连续；边界好了仍可能偶发追赶 |
| `jump_stitch0` | 与换点前速度同量级 | 若又≈0 且下一步很大 → skip0/C1 丢了 |
| `step1` | 与 `jump_stitch0` 接近、平滑过渡 | 突然接近 0 再飙高 → 边界 hitch |
| `raw_vs_guard` | 多为 0 或很小 | 很大 → 策略步长超 limit，guard 在砍 |
| `guard_sat` | 越低越好 | 很高 → 考虑略降速度或查轨迹 |

完整会话结束：

```text
[RTC] session stats: swaps=... guard_sat=...
```

### 5.3 用日志判「是不是又出现 hi→0→hi」

对 `schedule_joint_waypoint` 的相邻命令算每步 `max|Δq|`：

- 若在每次 `dt_swap` 附近反复出现 **大 → ≈0 → 大**：边界 hitch 回潮（检查 stitch 是否还在 skip0）。
- 若边界连续，但 blend 中后段突然顶满 limit：策略 `jump_raw0` 太大，调 `max_gw` / blend / soft_min，或接受策略本身跳变。

### 5.4 调参方向速查

| 体感 | 优先动 |
|------|--------|
| 每隔固定时间顿一下 / 刹一下 | stitch C1、skip0、是否重复下发 st0 |
| 换完后半段猛抽一下 | `jump_raw0`、`max_gw`、`RTC_BLEND_STEPS`、`soft_min` |
| 全程发硬、常顶速 | `max_joint_speed` / frequency、看 `guard_sat` |
| 越 RTC 越碎 | 是否 `(H+d)/2` 类过频 swap；`s_min` 是否过小 |

**不要**为了「加长 soft」而盲目把 `s_min` 加到 40+（soft 会更短）。要用 `soft_min` 垫 `exec_h`。

---

## 6. 涉及文件一览

| 文件 | 改动要点 |
|------|----------|
| `utils/kai0_rtc.py` | `stitch_chunk_to_state` 支持 `prev_velocity`（C1）+ 删除重复第 0 步；`soft_min` 垫高 `exec_h`；leftover hold-last（历史） |
| `scripts/eval_arx5_kai0_bimanual.py` | `prev_commanded`；`RTC_BLEND_STEPS=24`；`RTC_SOFT_MIN=20`；RTC 诊断日志；小批量 `schedule_batch≤8` |
| `data_local/kai0_eval/last_run.log` | 每次运行覆盖；抖的时候先看这个 |

辅助测试（无真机）：

- `scripts/test_kai0_rtc_smoke.py` — 连策略服务冒烟  
- `scripts/test_kai0_rtc_dry_loop.py` — 本地干跑 RTC 调度/swap

---

## 7. 一句话结论

> **抖的主因往往不是「RTC 算法坏了」，而是换 chunk 时位置钉死 + 重复下发同一姿态造成的一步刹车；C1 接速度并 skip 掉重复第 0 步后，体感会平滑很多。**  
> 策略的 `jump_raw0` 仍可能偏大，那是下一层问题，用 soft/`max_gw`/blend 缓和，不要和边界 hitch 混为一谈。
