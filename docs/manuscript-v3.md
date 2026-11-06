# Can Predictive Memory Reclamation Pay for Its Own CPU Cost?

**A Pre-registered Evaluation of a Distillation-Based Chunk Pool Policy**

> Status: v3 draft — §1–§9 complete.
> All citations reference verified BibTeX keys in `docs/related-work-matrix.md` §8.
> Scope: P1b-based C3 policy, BurstGPT trace-driven chunk pool simulation,
> WSL2 hardware (Intel Core Ultra 7 155H), T* = 2.6 s.

---

## 1. Introduction

Memory management in containerized cloud environments involves a persistent tension between two
costs. Returning idle memory to the operating system reduces pressure on a shared memory pool but
incurs a reacquisition penalty the next time the application demands that memory back.
Chunk pool allocators navigate this tension through fixed-schedule return policies:
jemalloc \cite{jemalloc} releases spans on a sigmoidal decay curve (decay\_ms = 10,000 ms by
default), and TCMalloc \cite{tcmalloc} releases at a fixed rate (TCMALLOC\_RELEASE\_RATE). Both
policies treat the return threshold as a universal time constant, independent of the
application's observed allocation pattern, and neither provides a formal accounting of the CPU
cost incurred by the return operation itself.

Predictive policies offer a natural improvement. If a model can estimate, for each idle memory
chunk, the probability that it will be reused within a given horizon, the policy can delay
returning high-reuse chunks and return low-reuse chunks promptly. Such a policy would reduce
expected reacquisition cost relative to a time-only rule while also reducing unnecessary holding.
Prior work demonstrates that this opportunity is real in related memory management domains:
LLAMA \cite{llama2020} learns heap allocation lifetimes to reduce allocation cost, and RecMG
\cite{recmg2025} uses LSTM+Attention predictions to guide embedding prefetching on tiered memory.

However, prediction is not free. Every decision to follow the learned path rather than the rule
path requires executing the model: computing features, performing table lookups, and applying
online scaling factors. These computations consume CPU time that would otherwise be available for
the application. In cloud containers operating under strict CPU quotas, this overhead is an
economic cost measured in the same currency as the memory savings. An AI-augmented return policy
is worthwhile only if the CPU value of its memory savings exceeds the CPU cost of obtaining those
savings.

Existing ML-based memory management systems acknowledge this tension without resolving it
formally. LLAMA \cite{llama2020} reports that inference takes 400 ns per call — 48$\times$
slower than the 8.3 ns direct path — and addresses this by amortizing the cost through caching;
no break-even condition is stated or tested. RecMG \cite{recmg2025}, the closest prior work in
this regard, explicitly measures a 92 µs inference cost and argues that pipelining hides it
behind embedding access latency. This engineering argument is not equivalent to a formal
break-even condition: "the cost is hidden" differs from "the memory benefit exceeds the cost."
Memory offloading systems — TMO \cite{tmo2022}, MEMTIS \cite{memtis2023} — optimize for
reclamation volume and either omit or impose an ad hoc bound on CPU overhead, without converting
memory savings to CPU-time units for comparison. To our knowledge, no prior work has stated an
explicit break-even condition $G_\mathrm{model} > 0$ for an ML memory reclamation policy,
pre-registered that condition before measurement, and reported the outcome.

This paper addresses the following question directly:

> *Can predictive memory reclamation pay for its own CPU cost?*

We instantiate this question through the C3 policy, a shadow-ledger switching mechanism that
combines an AI path (a P1b distilled lookup table derived from a 15-feature Cox survival gradient
boosted decision tree) with a rule-based fallback, selecting the lower-cost path over W = 300 s
windows. The P1b table covers three features — chunk height, usage level, and stack level — on a
65$^3$ grid ($\approx$268 KiB), and achieves a simulation-based memory savings value of
G1\_SMODEL\_C = 1,560.860 ms of CPU-equivalent benefit over the BurstGPT evaluation
workload \cite{burstgpt2025}.

We measure the actual incremental CPU overhead of the C3 decision path ($\Delta I$) over 420
evaluation windows across three measurement repetitions, using a C++ harness with
process-level CPU timers (CLOCK\_PROCESS\_CPUTIME\_ID) pinned to a single core (G2W3 format).

The main finding is negative. The measured average $\Delta I = 2{,}405.932$ ms exceeds the
simulation-based benefit G1\_SMODEL\_C = 1,560.860 ms by 845.072 ms. The net gain metric
$G_\mathrm{model}$ averages $-845.072$ ms. All three measurement repetitions fail the
pre-registered viability criterion $G_\mathrm{model} > 0$. Break-even sensitivity analysis
indicates that economic viability under the current hardware and workload would require either
reducing $\Delta I$ by at least 35.1\% or reducing $T^*$ by at least 31.8\%.

This paper makes the following contributions:

1. **Cost model and break-even criterion.** A two-component cost function $C = R + H/T^*$
   expressing memory management cost in CPU-time units, and a net gain metric
   $G_\mathrm{model} = c(\Delta R + \Delta H / T^*) - \Delta I$ with an explicit economic
   viability condition ($G_\mathrm{model} > 0$).

2. **Separation of prediction quality and execution cost.** A measurement design that evaluates
   the memory savings value (G1, simulation-based) and the CPU overhead (G2, hardware-measured)
   as independent, pre-registered phases, exposing the possibility that memory value is positive
   while net economic gain is negative.

3. **Complete measurement boundary.** A G2W3 timer scope that includes feature computation,
   table lookup, online scaling, and both policy paths within a single process-CPU timer,
   ensuring no overhead component escapes the measurement.

4. **Python–C++ parity verification.** A protocol confirming computational equivalence between
   the Python reference implementation and the C++ harness across 420 windows (420/420 PASS,
   zero deviation on all timing arrays).

5. **Pre-registered negative result.** The P1b-based C3 policy fails economic viability (G2
   FAIL) under the BurstGPT workload at T* = 2.6 s, with all three repetitions producing
   $G_\mathrm{model} < 0$ (range: $-768.157$ ms to $-948.249$ ms).

6. **Break-even sensitivity analysis.** Quantification of the design changes — $\Delta I$
   reduction, $T^*$ reduction, $\Delta H$ increase — that would bring future policy variants to
   economic viability.

---

## 2. Background and Related Work

We organize related work around the evaluation problem that motivates this study: measuring
whether an ML-based memory return policy creates more economic value than it consumes in CPU
time. We group prior work into five categories according to the management mechanism and the
degree to which CPU overhead is explicitly evaluated.

### 2.1 Memory Allocator Return Policies

Production chunk pool allocators return idle memory through rule-based schedules.
jemalloc \cite{jemalloc} releases spans on a sigmoidal decay curve, controlled by a single
decay\_ms parameter (10,000 ms by default). TCMalloc \cite{tcmalloc} releases at a fixed rate
(TCMALLOC\_RELEASE\_RATE). Both policies encode the same structural assumption: a universal
time-based threshold, rather than per-chunk predictions, is sufficient for the return decision.
Neither policy models the CPU cost of the return operation, nor does either explicitly trade off
return frequency against future reacquisition overhead.

The rule-based baseline in this study extends this structure with per-segment bin indexing (2
features: height and stack level) and an online recency correction via shadow ledger, while
retaining the same ML-free decision structure. This design serves as the economic comparator: a
predictive policy is justified only if it creates more value than this corrected rule.

### 2.2 ML-Based Memory Management

LLAMA \cite{llama2020} trains an offline LSTM to predict future heap allocation lifetimes and
sizes, steering requests toward pre-sized arenas to reduce fragmentation and allocation cost.
LLAMA explicitly measures that inference costs 400 ns per call, 48$\times$ slower than the 8.3 ns
direct path, and addresses this discrepancy through result caching to amortize the overhead
across repeated call sites. No formal condition is stated or tested for when the allocation
savings exceed the inference cost; the economic viability question is answered by an engineering
judgment ("amortize via caching") rather than a quantified comparison.

MEMTIS \cite{memtis2023} manages tiered memory (DRAM + slower tier) through histogram-based
access frequency tracking and targets a migration overhead below 3\% of application runtime.
This bound is a design constraint, not a break-even analysis: MEMTIS does not convert memory
savings to CPU-equivalent value or compare it against the management decision cost. The domain
(inter-tier page migration) also differs from chunk pool reclamation, as page migration targets
active access patterns rather than idle memory held in reserve.

### 2.3 Fleet-Level Memory Offloading and Page Reclamation

TMO \cite{tmo2022} achieves 20–32\% DRAM reduction across Meta's server fleet by monitoring
Linux Pressure Stall Information (PSI) and swapping cold pages to a compressed swap layer.
TMO uses no ML; reclamation decisions are entirely heuristic. The CPU cost of PSI polling and
reclamation is not separately reported; at fleet scale, it is assumed negligible relative to
DRAM savings. TMO does not provide a mechanism to test whether a prediction-based variant would
recover its inference cost, nor does it define an exchange rate between memory and CPU time.

A recent measurement study \cite{pagereclaim2025} characterizes the per-phase CPU cycle counts
of MGLRU and 2QLRU on Linux, providing fine-grained profiles of kernel-level page reclamation
mechanisms. This work measures overhead precisely but does not involve ML and does not model a
CPU-to-memory exchange rate. We adopt a similar approach of precise, boundary-defined CPU
accounting in our G2W3 measurement format.

### 2.4 ML-Based Cache Replacement and Embedding Prefetching

LeCaR \cite{lecar2018} blends LRU and LFU for cache line eviction through online reinforcement
learning, reporting the size of its regret table as a space overhead metric. CPU inference time
is not measured. The domain differs from heap return in a structurally important way: cache
eviction decisions are synchronous with the cache miss and cannot be deferred, whereas chunk
return decisions have flexibility in timing and can be batched. LeCaR evaluates replacement
quality (hit rate) but does not model the CPU cost of the blending decision.

RecMG \cite{recmg2025} is the closest prior work to ours. It applies LSTM+Attention to predict
DLRM embedding access patterns and guide prefetching on tiered memory, achieving up to 43\%
end-to-end speedup. RecMG explicitly reports that inference costs 92 µs per call and argues
that this cost is hidden by pipelining it with embedding access latency. This engineering
argument has merit in RecMG's setting — if the inference critical path lies off the application's
critical path, its wall-clock impact is zero — but it does not constitute a formal break-even
test for two reasons. First, the pipelining claim depends on hardware and workload concurrency
conditions that are not independently verified. Second, no formal condition of the form
"inference cost $<$ memory or latency benefit" is stated or checked. This work closes that gap
by defining $G_\mathrm{model} > 0$ as an explicit, pre-registered economic viability condition
and reporting the outcome, including a negative result.

### 2.5 CPU Overhead Evaluation in Memory Management Systems

KML \cite{kml2023} integrates lightweight ML models into the Linux block I/O scheduler,
reporting that model inference contributes less than 0.2\% CPU overhead. This is an overhead
percentage relative to total application runtime, not a break-even comparison against the
performance gain expressed in the same unit. KML provides a precedent for reporting kernel ML
overhead, but its domain (I/O scheduling) differs from memory reclamation, and the percentage
metric does not capture the opportunity cost of CPU time in a constrained container environment.

MemPerf \cite{memperf2023} provides an analytical profiler for diagnosing allocator-induced
performance slowdowns by decomposing memory access latency into allocator-attributable
components. MemPerf characterizes how to measure memory management overhead but does not propose
or evaluate an ML-based reclamation policy or a break-even condition.

### 2.6 Position of This Work

Across the works surveyed, we observe a consistent pattern: studies that report ML inference
costs do not test whether those costs are economically justified against their memory benefits
(RecMG, LLAMA), and studies that evaluate memory savings do not account for the CPU cost of the
management decision itself (TMO, MEMTIS). The CPU overhead constraints relevant to containerized
environments — where CPU time is a billed, quota-limited resource — are not reflected in the
evaluation frameworks of existing work.

**Table 1.** Comparison of related work on ML overhead evaluation and formal break-even analysis.

| Work | ML? | CPU overhead reported | Formal break-even? | Memory return domain |
|------|-----|-----------------------|--------------------|----------------------|
| jemalloc \cite{jemalloc} | No | No | No | Chunk pool (decay) |
| TCMalloc \cite{tcmalloc} | No | No | No | Chunk pool (rate) |
| LLAMA \cite{llama2020} | LSTM (offline) | 400 ns/call | No | Heap allocation |
| TMO \cite{tmo2022} | No (PSI) | Not reported | No | Fleet DRAM |
| MEMTIS \cite{memtis2023} | No (histogram) | \textless3\% (constraint) | No | Tiered memory |
| LeCaR \cite{lecar2018} | Online RL | Space only | No | Cache eviction |
| RecMG \cite{recmg2025} | LSTM+Attn | 92 µs/call | No | Tiered memory |
| KML \cite{kml2023} | Kernel-inline ML | \textless0.2\% | No | I/O scheduling |
| pagereclaim2025 \cite{pagereclaim2025} | No | Per-phase cycles | No | Kernel page reclaim |
| **This work** | P1b (distilled) | ΔI = 2,405.932 ms | **Yes** ($G_\mathrm{model} > 0$) | Chunk pool return |

---

## 3. Problem Formulation

### 3.1 Memory Return Cost Model

We model the cost of a chunk pool return policy over an evaluation window using a two-component
metric:

$$C = R + \frac{H}{T^*}$$

where $R$ (MiB) is the total memory returned to the operating system over the window, $H$
(MiB·s) is the total idle holding cost — the time integral of chunk memory held in an idle state
(allocated but not in active use) — and $T^*$ (s) is the break-even holding time.

The term $R$ captures the future reacquisition exposure: each MiB returned will cost $c$ µs of
CPU to reacquire if the application subsequently demands it, so a policy with high $R$ is
economically efficient only if most returned memory is not reacquired. The term $H/T^*$ captures
the opportunity cost of holding memory idle: memory held without purpose occupies container
quota that could be released to other workloads. The break-even holding time $T^*$ quantifies the
exchange rate: holding one MiB for $T^*$ seconds costs exactly as much as returning and
reacquiring it once.

The conversion factor $c$ (µs/MiB) ties the two cost components to a common unit. We measure $c$
empirically as the sum of the per-MiB return cost and the per-MiB reacquisition cost:

$$c = c_\mathrm{return} + c_\mathrm{reacq} = 25.6 + 256.2 = 281.8 \ \mu\mathrm{s/MiB}$$

$C$ is defined within the fluid model of the chunk pool: $R$ and $H$ are computed from the
allocator's return decisions and the chunk arrival/departure trace, without executing actual
system calls on live memory. The incremental decision cost is captured separately in the net
gain metric below.

### 3.2 Net Economic Gain

We evaluate whether policy C3 is economically viable relative to the rule-based baseline through
the net gain metric:

$$G_\mathrm{model} = c \cdot \left(\Delta R + \frac{\Delta H}{T^*}\right) - \Delta I$$

where:

- $\Delta R = R_\mathrm{rule} - R_\mathrm{C3}$ (MiB): the difference in returned memory; negative when C3 returns more memory than the rule.
- $\Delta H = H_\mathrm{rule} - H_\mathrm{C3}$ (MiB·s): the reduction in idle holding cost achieved by C3; positive when C3 avoids prolonged idle retention.
- $\Delta I = \mathrm{CPU}_{C3} - \mathrm{CPU}_\mathrm{rule}$ (ms): the incremental CPU time consumed by the C3 decision path relative to the rule path.

The term $c \cdot (\Delta R + \Delta H / T^*)$ — hereafter G1\_SMODEL\_C — is the
CPU-equivalent value of C3's memory management savings over rule, estimated from the fluid model.
$G_\mathrm{model} > 0$ holds if and only if this memory savings value strictly exceeds $\Delta I$.

$G_\mathrm{model}$ is a model-based economic estimate, not a realized CPU reduction or cloud
billing saving. Its validity is bounded by the fluid-model approximation for $R$ and $H$, the
representativeness of $c$ and $T^*$ measured on the evaluation platform, and the
representativeness of the BurstGPT workload \cite{burstgpt2025} for the target use case.

For analytical decomposition, we also report:

$$G_\mathrm{CPU} = c \cdot \Delta R - \Delta I$$

which excludes the holding-cost term and isolates whether C3's incremental return volume alone
justifies its CPU cost. We use $G_\mathrm{CPU}$ as a decomposition aid only; it cannot serve as
the primary viability criterion because a policy that never returns memory would minimize the
negative $\Delta R$ contribution while accumulating unbounded idle holding cost.

### 3.3 Sign Conventions and Current Values

Under the P1b-based C3 policy, the fluid model over 420 BurstGPT evaluation windows yields the
values summarized in Table 2.

**Table 2.** Key variables and their values under the P1b-based C3 policy.

| Variable | Value | Interpretation |
|---|---|---|
| $\Delta R = R_\mathrm{rule} - R_\mathrm{C3}$ | −906.0 MiB | C3 returns more memory than rule (R\_C3 > R\_rule) |
| $\Delta H = H_\mathrm{rule} - H_\mathrm{C3}$ | 16,756.72 MiB·s | Rule accumulates more idle holding than C3 |
| G1\_SMODEL\_C = $c(\Delta R + \Delta H/T^*)$ | 1,560.860 ms | Net memory benefit, CPU-equivalent (simulation) |
| $\Delta I$ (mean, 3 repetitions) | 2,405.932 ms | Incremental CPU cost of C3 path (measured) |
| $G_\mathrm{model}$ (mean) | **−845.072 ms** | Net economic gain: **negative (G2 FAIL)** |

The negative $\Delta R$ indicates that C3 is *more* aggressive in returning memory than the rule
policy (R\_C3 > R\_rule). The penalty from $c \cdot \Delta R < 0$ reflects that this additional
returned memory may incur reacquisition cost $c$ per MiB. The positive $\Delta H$ captures the
compensating benefit: by returning sooner, C3 avoids prolonged idle holding. The holding-cost
savings ($c \cdot \Delta H/T^* \approx 1{,}816$ ms) outweigh the reacquisition exposure
($c \cdot |\Delta R| \approx 255$ ms), yielding a positive G1\_SMODEL\_C = 1,560.860 ms.
However, this savings is insufficient to cover $\Delta I$ = 2,405.932 ms.

### 3.4 Two Evaluation Dimensions

The evaluation proceeds in two distinct phases that test complementary hypotheses.

**Phase G1 (simulation-based value estimation)** estimates G1\_SMODEL\_C = $c(\Delta R +
\Delta H/T^*)$ by running C3 and rule over the fluid model. G1 is an exploratory, pre-registered
selection phase: its purpose is to confirm that the memory savings value is large enough to
warrant CPU measurement (G1 pass criterion: G1\_SMODEL\_C $\geq$ 1.0 s), not to certify
economic viability.

**Phase G2 (CPU overhead measurement)** measures $\Delta I$ directly using a C++ harness with
process-level CPU timers across 420 evaluation windows and 3 measurement repetitions. The G2
pass criterion — $G_\mathrm{model} > 0$ for all three repetitions — is pre-registered and was
committed before the measurement was executed. The G2 result is the primary finding of this
paper.

A positive G1 outcome is a necessary but not sufficient condition for G2 viability: the memory
savings value must exist before it can cover $\Delta I$, but its existence does not guarantee
sufficiency. In this experiment, G1 is confirmed (G1\_SMODEL\_C = 1,560.860 ms > 1.0 s) and G2
is rejected ($G_\mathrm{model}$ averages $-845.072$ ms).

---

## 4. Methodology

### 4.1 Chunk Pool Fluid Model

We evaluate return policies using a *fluid model* of the chunk pool. The model tracks each chunk
as a discrete unit with an arrival timestamp, a departure timestamp (when the chunk is freed by
the application), and a decision event (when the policy decides to return or hold the chunk).
Given a return decision time $\tau$ for a chunk $i$ and the chunk's departure time $t_i^d$:
- If $\tau \leq t_i^d$ (the policy returns the chunk before the application actually frees it),
  the return event is logged and the chunk is added to $R$.
- After returning, if the application accesses the chunk again, the chunk must be reacquired from
  the OS, contributing $c \cdot (\text{chunk size})$ to the system's CPU cost.
- If $\tau > t_i^d$ (the chunk is held until the application frees it naturally), the idle
  holding time contributes to $H$.

The fluid model uses exact chunk arrival and departure times from the trace and operates
entirely in simulation: no actual system calls (madvise, mmap, munmap) are executed, no live
memory is allocated or freed, and $R$ and $H$ are accounting quantities computed from the trace
rather than measurements of actual RSS. The model is validated to match physical chunk pool
behavior in terms of return volumes under the same policy (§5.1 replay check), but does not
measure actual OS memory accounting or container resident set size.

### 4.2 Rule-Based Baseline Policy

The rule-based baseline assigns each idle chunk a return timeout $T_\mathrm{rule}$ based on a
2-feature bin index: chunk height $h$ (a size-related ordinal) and stack level $\ell$ (a
usage-pattern ordinal). The bins partition the chunk population into $G_r$ groups; within each
group, an online shadow ledger calibrates the timeout by comparing recent return decisions
against their outcomes. A fallback mechanism (run\_fallback3) handles edge cases where the
per-group estimate is unreliable by falling back to a baseline (single-group) timeout.

For the G2 measurement, the rule path's CPU time (CPU\_rule) covers: bin assignment
(compute\_grp\_r), online group-level timeout estimation (online\_group\_T for both baseline and
rule), the fallback selection (run\_fallback3), and the per-chunk cost accounting. File I/O,
window loading, and output writing are outside the timer scope.

### 4.3 C3 Policy: Structure and Decision Logic

C3 is a shadow-ledger switching policy that always executes both the rule path and the AI path,
then selects between their outputs based on the recent cost history of each. No approximation is
made: both paths produce full return-timeout arrays, and the selection is performed by a third
run\_fallback3 call comparing the two arrays against a window-based cost ledger.

The C3 decision structure for a single evaluation window is:

1. **Rule path** (shared with CPU\_rule): compute bin assignments, estimate online grouped
   timeouts for baseline and rule, apply rule fallback → produces $T_\mathrm{rb}$.

2. **AI path** (C3-exclusive): compute P1b features (compute\_p1b\_features:
   $\mathrm{pos}, D_0, m_D \to x_0, x_1, x_2$), perform lookup and $\kappa$-scaling
   (compute\_Tk: encode + table lookup + $\kappa$ scale), apply online $\kappa$ correction
   (online\_kappa\_T) → produces $T_\mathrm{ai}$. Apply AI fallback (run\_fallback3) →
   produces $T_\mathrm{ai,fb}$.

3. **C3 selection**: apply a third run\_fallback3 call comparing $T_\mathrm{ai,fb}$ and
   $T_\mathrm{rb}$ using the shadow ledger of recent $W = 300$ s costs → produces $T_\mathrm{c3}$.

Both paths run to completion on every window. The incremental CPU cost $\Delta I =
\mathrm{CPU}_{C3} - \mathrm{CPU}_\mathrm{rule}$ therefore includes all five C3-exclusive
operations: compute\_p1b\_features, compute\_Tk, online\_kappa\_T, run\_fallback3 (AI fallback),
and run\_fallback3 (C3 selection).

### 4.4 P1b Distilled Lookup Table

The AI path uses a P1b lookup table distilled from a 15-feature Cox proportional hazards
gradient boosted decision tree (GBDT). The distillation proceeds in two stages:

1. **Feature selection (G1)**: a pre-registered selection procedure identifies the 3-feature
   subset (height $x_0$, usage level $x_1$, stack level $x_2$) that minimizes average relative
   validation cost across 26 development schedules. The selection rule (minimize average relative
   cost; among ties, prefer higher G1\_SMODEL\_C) is committed before execution.

2. **Lookup table construction**: the GBDT is evaluated on a dense 65$^3$ grid over the three
   features, producing a table of 274,625 entries ($65 \times 65 \times 65 \approx 268$ KiB).
   The table value at index $(i_0, i_1, i_2)$ is the GBDT's predicted return timeout for a chunk
   with those encoded feature values. Integer-encoded feature values are computed by
   compute\_p1b\_features and used as direct indices.

3. **Online $\kappa$ correction**: the raw table lookup $\hat{T}_k$ is scaled by a per-window
   factor $\kappa$ maintained by online\_kappa\_T, which tracks the recent ratio of predicted
   timeouts to observed reuse times. This correction allows the lookup table to adapt to
   distribution shifts without retraining.

The P1b table was selected over the 2-feature P1 table (height omitted) because both passed the
G1 criterion (G1\_SMODEL\_C $\geq$ 1.0 s) and P1b achieved a higher value (P1b:
1,560.860 ms > P1: 1,483.856 ms). Both candidates were evaluated simultaneously on the same 420
development windows; the selection rule precedes the evaluation.

### 4.5 G2W3 Measurement Boundary

The G2W3 format defines the timer scope for measuring CPU\_rule and CPU\_C3. The timer is a
process-level CPU timer (CLOCK\_PROCESS\_CPUTIME\_ID) bracketing the full pass over all 420
windows for each path.

**CPU\_rule timer scope:**
- Ledger build (Ledger::build)
- Bin assignment: compute\_grp\_r (bin ID computation + unique remapping)
- Online grouped timeout estimation: online\_group\_T (baseline and rule variants)
- Fallback selection: run\_fallback3 (rule vs. baseline)
- Per-chunk cost accounting: compute\_costs

**CPU\_C3 timer scope:**
- All CPU\_rule operations (above)
- P1b feature computation: compute\_p1b\_features ($\mathrm{pos}, D_0, m_D \to x_0, x_1, x_2$)
- Table lookup and $\kappa$ scaling: compute\_Tk (encode + lookup + $\kappa$-scale)
- Online $\kappa$ correction: online\_kappa\_T
- AI fallback: run\_fallback3 ($T_\mathrm{ai}$ vs. $T_\mathrm{base}$)
- C3 selection: run\_fallback3 ($T_\mathrm{ai,fb}$ vs. $T_\mathrm{rb}$)
- Per-chunk cost accounting: compute\_costs

Operations outside both timers: file I/O, memory loading of windows and model, and output
writing. This boundary ensures that $\Delta I = \mathrm{CPU}_{C3} - \mathrm{CPU}_\mathrm{rule}$
captures exactly the incremental CPU cost of the AI-augmented path relative to the rule path.

**Table 3.** G2W3 timer boundary summary.

| Operation | CPU\_rule? | CPU\_C3? | Incremental? |
|---|---|---|---|
| Ledger build | ✓ | ✓ | No |
| compute\_grp\_r | ✓ | ✓ | No |
| online\_group\_T (base + rule) | ✓ | ✓ | No |
| run\_fallback3 (rule) | ✓ | ✓ | No |
| compute\_p1b\_features | ✗ | ✓ | **Yes** |
| compute\_Tk (lookup + κ) | ✗ | ✓ | **Yes** |
| online\_kappa\_T | ✗ | ✓ | **Yes** |
| run\_fallback3 (AI fallback) | ✗ | ✓ | **Yes** |
| run\_fallback3 (C3 selection) | ✗ | ✓ | **Yes** |
| File I/O, memory loading | ✗ | ✗ | — |

### 4.6 Python–C++ Parity Verification

Before the official CPU measurement, we verify that the C++ harness and the Python reference
implementation (g2\_prepare.py) produce numerically identical results for all 420 evaluation
windows. The parity check (g2\_parity.py) compares six timing arrays per window:
$T_\mathrm{base}$, $T_\mathrm{rule}$, $T_\mathrm{rb}$, $T_\mathrm{ai}$,
$T_\mathrm{ai,fb}$, $T_\mathrm{c3}$. The comparison uses an absolute tolerance of
$T_\mathrm{TOL} = 0.0$ for timing arrays (exact equality) and $1 \times 10^{-9}$ for $R$ and $H$
scalar values (floating-point rounding only).

Across 420 windows and 7,671,239 slice-level comparisons, parity is confirmed: 420/420 PASS,
zero failures. This verification ensures that the C++ harness faithfully implements the same
policy logic as the Python reference, so that G2 CPU measurements reflect the same computational
procedure whose memory management behavior is evaluated in the fluid model.

---

## 5. Experimental Setup

### 5.1 Evaluation Workload

The evaluation workload is derived from BurstGPT \cite{burstgpt2025}, an LLM serving trace
dataset containing timestamped request records collected from Azure OpenAI over 213 days.
Each record provides: timestamp, session ID, elapsed time, model name, request token count,
response token count, and total token count. BurstGPT is a request-level serving trace;
it is not a memory allocation log and does not directly record allocator-level events.

The chunk pool workload is constructed from these LLM serving traces through a preprocessing
pipeline that maps serving request patterns to chunk pool allocation and deallocation events.
The resulting workload captures the bursty, session-structured nature of LLM inference
workloads. This study uses BurstGPT v2 and v3 traces spanning 40 development days and 41 sealed
evaluation days.

**Evaluation windows.** Traces are partitioned into non-overlapping *windows* of fixed duration.
Each window contains a contiguous segment of the trace with its associated chunk pool events.
Windows where total activity falls below a qualification threshold are excluded. The final
evaluation set for G1 and G2 consists of 420 qualified windows spanning 40 development days.
The v2 sealed evaluation used 405 windows spanning 41 days (a subset of the development
windows that met the v2 qualification criteria). The v1 sealed evaluation used 53 windows
(39 from BurstGPT, 14 from an Azure 2024 trace dataset).

**Replay check.** A chunk pool replay over 160 windows confirmed that actual return volumes
match the fluid model predictions. The maximum discrepancy between modeled and actual RSS
was 0.2 MiB.

### 5.2 Decisions

Across the 420 G2 evaluation windows, the total number of *test events* (TE) — chunk-level
return decisions within the evaluation time range of each window — is 2,003,785. This count is
identical for the rule and C3 paths (confirmed by the C++ harness), reflecting that both paths
process the same set of decisions.

### 5.3 Hardware and Software Environment

**Table 4.** Evaluation hardware and software environment.

| Item | Value |
|---|---|
| CPU | Intel(R) Core(TM) Ultra 7 155H |
| OS / kernel | Linux 6.6.87.2-microsoft-standard-WSL2 |
| Virtualization | WSL2 (Windows Subsystem for Linux, version 2) |
| C++ compiler | g++ (Ubuntu 13.3.0-6ubuntu2\textasciitilde24.04.1) 13.3.0 |
| Python | 3.12.3 (GCC 13.3.0) |
| CPU affinity | taskset -c 1 (single core, core 1) |
| CPU timer | CLOCK\_PROCESS\_CPUTIME\_ID |
| OMP\_NUM\_THREADS | 1 (single-threaded) |
| OPENBLAS\_NUM\_THREADS | 1 |
| Git commit | 5c07328c2c2fc30dc058453f4fcc067d00f503c1 |

The WSL2 environment provides process isolation and Linux system call semantics under Windows.
All CPU measurements use process-CPU time, which counts only time the measured process is
scheduled and executing on the CPU, excluding time spent waiting for I/O or OS interrupts.

### 5.4 G2W3 Measurement Protocol

The official measurement follows the G2W3 protocol:

1. **Warmup** (1 pass, discarded): run\_cpu\_rule and run\_cpu\_c3 each once over all 420
   windows to bring instruction caches and branch predictors to a stable state.

2. **Iteration 1** (order: rule\_c3): run rule path, then C3 path.

3. **Iteration 2** (order: c3\_rule): run C3 path, then rule path.

4. **Iteration 3** (order: rule\_c3): run rule path, then C3 path.

The alternating execution order (rule\_c3, c3\_rule, rule\_c3) is designed to detect execution-order
effects: if prior C3 execution affects subsequent rule measurement (e.g., through cache warming),
the c3\_rule and rule\_c3 iterations would differ. All three official iterations are used in the
primary analysis; $\Delta I_i = \mathrm{CPU}_{C3,i} - \mathrm{CPU}_{\mathrm{rule},i}$ is
computed independently for each iteration $i$.

**Determinism check.** For each iteration, the C++ harness computes 64-bit checksums of the
complete rule-path and C3-path output arrays. Matching checksums across iterations confirm that
the same return decisions are produced regardless of execution order and system state.

---

## 6. Results

We report results in the order of the evaluation pipeline: v1 sealed experiment ($§6.1$),
policy development ($§6.2$), T* calibration ($§6.3$), v2 sealed experiment ($§6.4$), Python CPU
overhead ($§6.5$), placebo analysis ($§6.6$), simulation-based economic value ($§6.7$), official
CPU measurement ($§6.8$), and break-even sensitivity ($§6.9$).

**Notation.** We distinguish evidence types: *(sim)* = fluid model simulation; *(meas)* = direct
hardware measurement; *(calc)* = derived from simulation or measurement values by formula.

### 6.1 v1 Sealed Experiment

**Question.** Does the AI survival policy reduce memory return cost relative to a fixed-online
baseline (H1) and a rule-online-fallback baseline (H2)?

**Setup.** Pre-registered protocol v1.1 (commit `0a3fc73`, tag `prereg-v1.1`), executed
2026-10-24. Evaluation set R: 53 windows (39 BurstGPT, 14 Azure 2024). Evaluation T* = 1 s.

**Results (sim, sealed v1).** Table 5 summarizes the v1 hypothesis outcomes.

**Table 5.** v1 sealed experiment results (simulation, T* = 1 s, 53 evaluation windows).

| Hypothesis | Comparison | Win/Tie/Loss | p-value | Effect (median %p) | Verdict |
|---|---|---|---|---|---|
| H1 | AI vs fixed-online | 37/12/4 | $1.0 \times 10^{-7}$ | −7.68 | **PASS** |
| H2 | AI vs rule-online+fallback | 9/14/30 | 0.0011 | +1.16 | **FAIL** |

**Interpretation.** H1 confirms that the AI policy outperforms a universal fixed threshold.
However, H2 reveals that a 2-feature rule policy with the same online correction mechanism
achieves lower simulation cost than the AI policy at T* = 1 s (rule wins 30/53 windows,
AI wins 9). The H2 failure motivates the development of C3: rather than relying on the AI path
alone, C3 selects between AI and rule using a shadow ledger.

**Post-hoc model quality check.** During v2 preparation, three causal leakage defects were
identified in the v1 fluid model: (i) chunk position features used future-arrival sizes at
decision time, (ii) shadow ledger rounding introduced non-deterministic comparisons, and (iii)
row ordering in chunk arrays correlated with future reuse times. A corrected chunk-model
recomputation on the same R windows preserved H1 and H2 direction and significance
(H1: 36/14/3, $p = 3.6 \times 10^{-8}$; H2: 9/9/35, $p = 1.1 \times 10^{-4}$). The
corrected model is not used for judgment; it confirms that the v1 qualitative conclusions
are robust to the identified defects.

### 6.2 C3 Policy Selection

**Question.** Which policy variant among C1–C4 (AI-rule switching candidates) minimizes
simulation cost on v1 development windows while satisfying the retreat constraint?

**Setup.** v1 sealed windows were converted to development data after the v1 judgment. Policy
candidates were evaluated under the corrected chunk model at T* = 0.3, 1, 3 s. Elimination
criterion: retreat ratio (cost increase relative to fixed-online) > 10\%.

**Results (sim, development).** C1 (retreat 20.4\%), C2 (11.1\%), and C4 (13.0\%) were
eliminated. C3 had a maximum retreat ratio of 9.3\% and satisfied the win-majority condition at
multiple T* values. C3 was selected as the v2 candidate policy.

C3's switching structure is designed to capture AI gains where the AI path outperforms rule
while reverting to rule when the AI path underperforms, over W = 300 s windows.

### 6.3 T* and c Calibration

**Question.** What are the per-MiB return cost $c$ and the break-even holding time $T^*$ on the
evaluation platform?

**Setup** (meas). Measurement conditions: 1 MiB chunks, 4 KiB pages, MADV\_DONTNEED, batch
size 16, single core (core 1), 300 repetitions per trial.

**Results** (meas). Table 6 reports the calibration results.

**Table 6.** Return/reacquisition cost and break-even holding time.

| Quantity | Value | Type |
|---|---|---|
| $c_\mathrm{return}$ (per-MiB return cost) | 25.6 µs/MiB | meas |
| $c_\mathrm{reacq}$ (per-MiB reacquisition cost) | 256.2 µs/MiB | meas |
| $c = c_\mathrm{return} + c_\mathrm{reacq}$ | 281.8 µs/MiB | calc |
| Effective T* (trials 1, 2, 3) | 2.61, 2.64, 2.63 s | meas |
| **T* (median, 2 sig. figs.)** | **2.6 s** | meas |

Under Transparent Huge Pages (THP) enabled with 2 MiB chunks, the effective T* drops to
approximately 1.0 s. This configuration was not used in the primary evaluation; the standard
1 MiB, 4 KiB-page configuration gives T* = 2.6 s.

The replay check over 160 windows confirmed that actual return volumes match fluid model values,
with a maximum deviation of 0.2 MiB between modeled and measured resident memory.

### 6.4 v2 Sealed Experiment

**Question.** Does C3 reduce simulation cost relative to the rule baseline (H1v2), relative to
the fixed-online baseline (H2v2), and in synthetic confirmation windows (H3v2)?

**Setup.** Pre-registered protocol v2 (tag `prereg-v2`), executed 2026-10-26 (commit
`7f0b0960`). Evaluation: BurstGPT 41 days, 405 qualified windows. Judgment T* = 2.6 s.

**Results (sim, sealed v2).** Table 7 summarizes the v2 hypothesis outcomes.

**Table 7.** v2 sealed experiment results (simulation, T* = 2.6 s, 405 evaluation windows).

| Hypothesis | Comparison | Measured effect | Threshold | Verdict |
|---|---|---|---|---|
| H1v2 | C3 − rule (41-day total cost) | −2.34 %p | < −0.5 %p | **PASS** |
| H2v2 | C3 − fixed-online (retreat ≤ 10\%) | −9.02 %p | < −0.5 %p | **PASS** |
| H3v2 | C3 − rule, synthetic schedules (Holm) | 27/9/4, $p = 3.4 \times 10^{-5}$ | Holm < 0.05 | **PASS** |

Absolute total simulation costs (41 days): C3 = 767,024.9 MiB; rule = 786,730.4 MiB;
fixed-online = 843,054.0 MiB; oracle (lower bound) = 625,272.5 MiB.

**Interpretation.** All three v2 hypotheses are confirmed. C3 achieves a 2.34 percentage-point
reduction in total simulation cost relative to rule, and a 9.02 percentage-point reduction
relative to fixed-online. These results establish that C3 creates positive memory management
value in the fluid model at T* = 2.6 s. They do not, by themselves, establish economic
viability: the CPU cost of obtaining this value is measured separately in G2.

### 6.5 Task A: Python CPU Overhead

**Question.** Does the Python scikit-learn implementation of the full AI pipeline fit within the
per-window CPU budget for two training configurations?

**Setup** (meas, development). Platform: WSL2, Intel Core Ultra 7 155H, Python 3.12.3,
scikit-learn 1.9.1, single-threaded, single core. 420 development windows, T* = 2.6 s.
Measurement instrument: measure\_a\_cpu.py (commit `a071c92`), 2026-10-31.

Two training configurations were evaluated:

- *Within C3*: the model is retrained on each window's history before inference.
- *Pre-trained C3*: a fixed model trained on 26 development schedules is applied to 420 windows.

**Results (meas, development).** Table 8 reports the Task A outcomes.

**Table 8.** Task A: Python AI pipeline CPU cost versus per-window budget
(development data, Python implementation).

| Configuration | Metric | Measured | Budget | Ratio | Verdict |
|---|---|---|---|---|---|
| Within C3 | Per-window training time | 3,443–3,489 ms | $2 \times S \times c$ = 27.55 ms | 125–127$\times$ | **FAIL** |
| Within C3 | Batch inference total | 47.34–48.30 s | $S \times c$ = 5.785 s | 8.18–8.35$\times$ | **FAIL** |
| Pre-trained C3 | Batch inference total | 51.73–51.92 s | $S_\mathrm{pre} \times c$ = 3.350 s | 15.4–15.5$\times$ | **FAIL** |

Here $S$ denotes the simulation-based gain for the within-C3 configuration, and
$S_\mathrm{pre}$ denotes the gain for the pre-trained configuration, converted to CPU-time
equivalents using $c = 281.8$ µs/MiB.

**Per-decision summary.** The per-decision gain is 2.89 µs (within C3) or 1.67 µs (pre-trained
C3), while batch inference costs 23.6–24.1 µs/decision (within C3) and 25.8–25.9 µs/decision
(pre-trained C3), respectively 8.2–8.3$\times$ and 15.4–15.5$\times$ the per-decision gain.

**Scope note.** Task A measures the Python 3.12 / scikit-learn 1.9.1 single-threaded
implementation. These results do not generalize to optimized C++ implementations; the C++
P1b-based measurement is reported separately in §6.8.

### 6.6 Task B: Placebo Analysis

**Question.** How much of C3's simulation gain originates from per-chunk prediction information,
versus timing, prediction distribution, and switching structure?

**Setup** (sim, post-hoc, development). The placebo analysis was triggered by the Task A
double-failure outcome and was not part of the original pre-registered protocol. It is a
post-hoc exploratory analysis on development data. Platform: Intel Xeon 2.10 GHz, Python 3.13.18.

Three placebo variants replace the AI prediction with structurally equivalent but informationally
degraded predictions:
- *P\_shuffle*: within each 30 s segment, per-chunk predictions $S_k$ are row-shuffled
  (breaking per-chunk correspondence while preserving segment-level distribution).
- *P\_draw*: predictions are drawn from the prior 30 s segment (breaking temporal correspondence).
- *P\_label*: a model retrained on permuted labels (removing predictive signal).

**Results (sim, development).** Table 9 reports the additive decomposition of the total gain
$\Delta d_\mathrm{total} = -2.497$ %p across 5 random seeds.

**Table 9.** Task B placebo decomposition of C3's simulation gain (development data).

| Component | Measured range (5 seeds, %p) | Fraction of total |
|---|---|---|
| $C3 - P_\mathrm{shuffle}$ (per-chunk correspondence) | −2.492 to −2.482 | 99.4–99.8\% |
| $P_\mathrm{shuffle} - P_\mathrm{draw}$ (temporal correspondence) | −0.051 to −0.038 | ≈1.5–2\% |
| $P_\mathrm{draw} - P_\mathrm{label}$ (distributional signal) | +0.012 to +0.043 | ≈−0.5–1.7\% |
| $P_\mathrm{label} - \mathrm{rule}$ (switching structure) | −0.010 to +0.015 | ≈−0.4–0.6\% |
| Interaction term (C3 switching absorbs placebo losses) | +5.429 to +5.447 | n/a |

The dominant component (99.4–99.8\%) is per-chunk prediction correspondence: when the
association between $S_k$ and each chunk is broken by shuffling within a 30 s segment, the
simulation gain nearly vanishes. Temporal correspondence, distributional properties, and
switching structure each contribute less than ±2\% of the total gain.

**Interpretation.** C3's simulation gain originates primarily from the per-chunk predictive
information in the AI path, not from artifacts of the switching structure or aggregate prediction
statistics. The economic shortfall (§6.8) therefore reflects a situation where the right
information is available but too expensive to obtain with the current implementation: the
per-chunk information value is approximately 2.49 %p (≈ 1,560.860 ms CPU-equivalent), while the
CPU cost of obtaining it via the P1b path is 2,405.932 ms.

### 6.7 G1: Simulation-Based Economic Value

**Question.** Which P1b feature subset (P1 or P1b) achieves the highest simulation-based
economic value G1\_SMODEL\_C, and does this value exceed the G1 passage threshold?

**Setup** (sim, development). Pre-registered feature selection plan v4 (commit `3912a9a`,
2026-11-04). Candidates: P1 (usage level, stack level; $65^2$ grid) and P1b (height, usage level,
stack level; $65^3$ grid). Selection criterion: both must pass G1\_SMODEL\_C $\geq$ 1.0 s;
if both pass, select the candidate with the higher G1\_SMODEL\_C.

**Results (sim, development).** Table 10 reports the G1 candidate evaluation.

**Table 10.** G1 candidate comparison over 420 development windows.

| Candidate | Features | Grid | G1\_SMODEL\_C | G1 verdict |
|---|---|---|---|---|
| P1 | Usage level, stack level | $65^2$ (64 KiB) | 1,483.856 ms | PASS |
| **P1b** | **Height, usage level, stack level** | $65^3$ (268 KiB) | **1,560.860 ms** | **PASS** |

Both candidates pass the G1 threshold (1.0 s). P1b is selected because
$1{,}560.860 > 1{,}483.856$ ms. The selection is applied before G2 measurement.

The G1 values decompose as: for P1b, $\Delta R = -906.0$ MiB and
$\Delta H = 16{,}756.72$ MiB·s, giving
$c(\Delta R + \Delta H/T^*) = 281.8 \times (-906.0 + 6{,}444.9) = 281.8 \times 5{,}538.9 \approx 1{,}560.860$ ms.

### 6.8 G2W3: Official CPU Measurement

**Question.** Does the P1b-based C3 policy satisfy the economic viability criterion
$G_\mathrm{model} > 0$ for all three measurement repetitions?

**Setup** (meas, official). Git commit `5c07328c`, g++ 13.3.0, WSL2, taskset -c 1,
CLOCK\_PROCESS\_CPUTIME\_ID, OMP\_NUM\_THREADS=1, 420 windows, 2026-11-05. P1b model SHA-256:
`35338f0f...`. C++ harness SHA-256: `a8eec698...`.

**Results (meas, official).** Table 11 reports the three official measurement iterations.

**Table 11.** G2W3 official measurement results (C++ harness, 420 windows × 3 repetitions).

| Iter. | Order | CPU\_rule (s) | CPU\_C3 (s) | $\Delta I$ (ms) | $G_\mathrm{model}$ (ms) | $G_\mathrm{CPU}$ (ms) |
|---|---|---|---|---|---|---|
| 1 | rule\_c3 | 17.0735 | 19.4025 | 2,329.017 | −768.157 | −2,584.328 |
| 2 | c3\_rule | 17.0810 | 19.4607 | 2,379.671 | −818.811 | −2,634.982 |
| 3 | rule\_c3 | 16.9037 | 19.4128 | 2,509.109 | −948.249 | −2,764.420 |
| **Mean** | | 17.0194 | 19.4253 | **2,405.932** | **−845.072** | **−2,661.243** |

Population standard deviation (n = 3): $\sigma_{\Delta I} = 75.831$ ms;
sample standard deviation: $s_{\Delta I} = 92.873$ ms.

**Determinism check.** checksum\_rule = `b2bb4067b5354578`, checksum\_c3 = `ba240bd4f8369334`,
identical across all three iterations. The same return decisions are produced in every repetition.

**G2 judgment.** The pass condition is $G_\mathrm{model} > 0$ for all three iterations. All
three iterations produce $G_\mathrm{model} < 0$ (range: $-768.157$ ms to $-948.249$ ms).

> **G2 final verdict: FAIL.**

The mean $\Delta I = 2{,}405.932$ ms exceeds G1\_SMODEL\_C = 1,560.860 ms by 845.072 ms. The
per-decision $\Delta I$ averages 1.201 µs/decision (2,405.932 ms / 2,003,785 decisions), while
the per-decision simulation value is 0.779 µs/decision (1,560.860 ms / 2,003,785 decisions).

The minimum $\Delta I$ across all three iterations (2,329.017 ms, iteration 1) still exceeds
G1\_SMODEL\_C by 768.157 ms, confirming that the economic shortfall is not attributable to a
single high-overhead iteration. The range across iterations (180.092 ms) is substantially
smaller than the mean shortfall (845.072 ms), ruling out measurement instability as a confound.

$G_\mathrm{CPU}$ averages $-2{,}661.243$ ms, substantially more negative than $G_\mathrm{model}$.
This decomposition indicates that the $\Delta H / T^*$ term contributes positively
(≈1,816 ms), partially offsetting the combined negative contributions of $c \cdot \Delta R$
(≈−255 ms) and $\Delta I$ (−2,406 ms). The economic shortfall arises primarily from $\Delta I$
exceeding G1\_SMODEL\_C, not from an adverse $\Delta R$.

### 6.9 Break-Even Sensitivity Analysis

**Question.** What single-variable changes would bring $G_\mathrm{model}$ to zero, and how large
are those changes relative to current values?

**Setup** (calc). Baseline: mean $\Delta I = 2{,}405.932$ ms, G1\_SMODEL\_C = 1,560.860 ms,
$c = 281.8$ µs/MiB, $T^* = 2.6$ s, $\Delta R = -906.0$ MiB, $\Delta H = 16{,}756.72$ MiB·s.

**Results (calc).** Table 12 reports the single-variable break-even conditions.

**Table 12.** Single-variable break-even conditions for $G_\mathrm{model} > 0$
(mean $\Delta I$ baseline; strict inequalities).

| Variable | Current value | Break-even condition ($G_\mathrm{model} > 0$) | Required change |
|---|---|---|---|
| $\Delta I$ | 2,405.932 ms | $< 1{,}560.860$ ms | −35.1\% |
| $T^*$ | 2.6 s | $< 1{.}774$ s | −31.8\% |
| $c$ | 281.8 µs/MiB | $> 434{.}371$ µs/MiB | +54.1\% |
| $\Delta H$ | 16,756.72 MiB·s | $> 24{,}553{.}699$ MiB·s | +46.5\% |
| $\Delta R$ | −906.0 MiB | $> {+}2{,}092{.}837$ MiB | Sign reversal (+2,999 MiB) |

Boundary values are the points where $G_\mathrm{model} = 0$; the strict conditions require
exceeding them.

**Interpretation of individual paths:**

- *$\Delta I$ path (−35.1\%)*: The C3 path would need to process 420 windows in
  $\approx$1,560 ms instead of $\approx$2,406 ms. The five incremental operations — 
  compute\_p1b\_features, compute\_Tk, online\_kappa\_T, and two run\_fallback3 calls — collectively
  account for this overhead. Their individual contributions are not separately measured in
  the current implementation.

- *$T^*$ path (−31.8\%)*: T* = 1.774 s lies between the standard measurement (2.6 s) and
  the THP-enabled 2 MiB configuration (≈1.0 s). It does not correspond to any measured hardware
  configuration. T* is fixed by the pre-registered v2 protocol and cannot be retroactively
  adjusted to change the G2 judgment.

- *$\Delta R$ path (sign reversal)*: Break-even requires C3 to return *less* memory than rule
  ($\Delta R > 0$), which is the opposite of its current behavior ($\Delta R = -906$ MiB,
  C3 returns more). Achieving this would require a qualitatively different reclamation strategy.

- *Composite conditions*: When multiple variables improve simultaneously, the required
  per-variable improvement is smaller. For example, a 20\% reduction in $\Delta I$ combined with
  a 15\% reduction in $T^*$ would bring $G_\mathrm{model}$ to approximately zero; the exact
  threshold depends on the starting point for each variable.

The sensitivity analysis does not constitute a plan or prediction for future policy variants;
it quantifies the design space for future investigation.

---

## 7. Discussion

### 7.1 Policy Quality and Economic Viability

The evaluation addresses two distinct questions through separate experimental phases. The v2
sealed experiment (§6.4) tests whether C3 makes better memory decisions than the rule baseline
under the fluid model — a question of *policy quality*. The G2 measurement (§6.8) tests whether
those improved decisions create more economic value than they consume in CPU time — a question of
*economic viability*. The two phases can produce opposite answers, and in this evaluation they do.

All three v2 hypotheses are confirmed: H1v2 (C3 − rule: −2.34 %p, PASS), H2v2
(C3 − fixed-online: −9.02 %p, PASS), and H3v2 (synthetic schedules: 27/9/4,
$p = 3.4 \times 10^{-5}$, PASS). These results establish that C3 reduces fluid-model simulation
cost below both baselines at T* = 2.6 s, on BurstGPT 41-day evaluation data.

The G2 measurement finds mean $\Delta I = 2{,}405.932$ ms against G1\_SMODEL\_C = 1,560.860 ms,
yielding mean $G_\mathrm{model} = -845.072$ ms. All three official measurement repetitions
independently fail the economic viability criterion $G_\mathrm{model} > 0$. The parity
verification (420/420 PASS, zero deviation) ensures that the same policy logic producing the v2
quality improvements is exactly what the C++ harness measures for CPU overhead.

The relationship between the two phases is directional but not sufficient: a positive G1 outcome
(G1\_SMODEL\_C $\geq$ 1.0 s) is necessary for G2 viability because the economic value must
exist before it can cover $\Delta I$. Its existence does not guarantee sufficiency. This
asymmetry — policy quality is positive but insufficient to cover execution cost — is the central
structural finding of this evaluation.

### 7.2 Decomposition of the Economic Outcome

The net gain $G_\mathrm{model} = c(\Delta R + \Delta H/T^*) - \Delta I$ has three additive
components under the confirmed parameter values. Table 13 presents the per-term decomposition.

**Table 13.** Term-by-term decomposition of $G_\mathrm{model}$.

| Term | Computation | Value |
|------|-------------|-------|
| $c \cdot \Delta R$ | $281.8 \times (-906.0)$ µs | $-255.311$ ms |
| $c \cdot \Delta H / T^*$ | $281.8 \times 16{,}756.72 / 2.6$ µs | $+1{,}816.171$ ms |
| G1\_SMODEL\_C = $c(\Delta R + \Delta H/T^*)$ | $-255.311 + 1{,}816.171$ | $+1{,}560.860$ ms |
| $\Delta I$ (mean, 3 repetitions, meas) | measured | $+2{,}405.932$ ms |
| $G_\mathrm{model}$ (mean) | $1{,}560.860 - 2{,}405.932$ | $-845.072$ ms |

The sign of $\Delta R = -906.0$ MiB requires careful interpretation. A negative $\Delta R$
means C3 returns *more* memory than the rule policy ($R_{C3} > R_\mathrm{rule}$ by 906 MiB).
The $c \cdot \Delta R$ term is therefore negative (−255.311 ms): the additional returned memory
creates reacquisition exposure worth 255.311 ms in CPU units if that memory is subsequently
demanded. This is not evidence of a policy defect.

The reason G1\_SMODEL\_C is positive despite negative $c \cdot \Delta R$ is that C3's more
aggressive return behavior simultaneously reduces idle holding time by
$\Delta H = +16{,}756.72$ MiB·s, contributing $+1{,}816.171$ ms. This holding-cost savings
outweighs the reacquisition exposure by a factor of approximately 7:1
($1{,}816 / 255 \approx 7.1$), yielding a net positive memory savings value of
$+1{,}560.860$ ms. The economic shortfall arises not from an adverse memory management
outcome, but from $\Delta I = 2{,}405.932$ ms exceeding this savings value by 845.072 ms.

The placebo analysis (§6.6) provides additional context: 99.4–99.8\% of C3's simulation gain
originates from per-chunk prediction correspondence. The economic shortfall therefore reflects
a situation where the right per-chunk information is demonstrably useful in the fluid model,
but the current implementation obtains that information at a cost that exceeds its value.

### 7.3 CPU Cost of the Prediction and Execution Path

The incremental CPU overhead $\Delta I = \mathrm{CPU}_{C3} - \mathrm{CPU}_\mathrm{rule}$
captures the aggregate cost of all operations present in the CPU\_C3 timer scope but absent
from the CPU\_rule scope. Five categories of operations contribute to $\Delta I$:

1. **Feature computation** (compute\_p1b\_features): maps per-chunk position and arrival
   statistics ($\mathrm{pos}$, $D_0$, $m_D$) to three integer-encoded feature values
   ($x_0$, $x_1$, $x_2$) used as lookup indices into the P1b grid.

2. **Table lookup and scaling** (compute\_Tk): encodes the three feature values, retrieves
   the corresponding timeout from the 65$^3$ P1b grid ($\approx$268 KiB), and applies the
   per-window $\kappa$ scale factor.

3. **Online $\kappa$ correction** (online\_kappa\_T): maintains and updates the window-level
   $\kappa$ estimate tracking the ratio of predicted timeouts to observed reuse times.

4. **AI path fallback** (run\_fallback3): compares $T_\mathrm{ai}$ against $T_\mathrm{base}$
   using the shadow ledger; selects $T_\mathrm{ai,fb}$.

5. **C3 selection** (run\_fallback3): compares $T_\mathrm{ai,fb}$ against $T_\mathrm{rb}$
   using the recent $W = 300$ s cost ledger; produces the final C3 decision.

Operations shared between both paths — Ledger::build, compute\_grp\_r, online\_group\_T,
run\_fallback3 (rule vs. baseline), and compute\_costs — appear in both CPU\_rule and CPU\_C3
timers and cancel in the difference. They do not contribute to $\Delta I$.

The G2W3 measurement records only the aggregate $\Delta I$; individual function contributions
within the incremental path are not separately instrumented or measured in this evaluation.
The decomposition above identifies which operations structurally contribute to $\Delta I$,
but their relative CPU shares cannot be determined from the available data.

### 7.4 Break-Even Analysis

The break-even conditions in §6.9 are single-variable sensitivity estimates: each condition
fixes all other parameters at their current values and solves $G_\mathrm{model} = 0$ for the
variable of interest. The boundary value is the point where $G_\mathrm{model} = 0$; the strict
viability condition $G_\mathrm{model} > 0$ requires strictly exceeding it.

| Variable | Current value | Required for $G_\mathrm{model} > 0$ | Required change |
|----------|---------------|--------------------------------------|-----------------|
| $\Delta I$ | 2,405.932 ms | $< 1{,}560.860$ ms | −35.1\% |
| $T^*$ | 2.6 s | $< 1{.}774$ s | −31.8\% |
| $c$ | 281.8 µs/MiB | $> 434{.}371$ µs/MiB | +54.1\% |
| $\Delta H$ | 16,756.72 MiB·s | $> 24{,}553{.}699$ MiB·s | +46.5\% |
| $\Delta R$ | −906.0 MiB | $> {+}2{,}092{.}837$ MiB | Sign reversal (+2,999 MiB) |

The $\Delta R$ break-even condition deserves particular attention. Currently
$\Delta R = -906.0$ MiB (C3 returns 906 MiB more than rule). Break-even requires
$\Delta R > +2{,}092.837$ MiB, meaning C3 would need to return approximately 2,999 MiB *less*
than rule — a sign reversal. This represents a qualitatively different reclamation strategy from
the current C3 policy.

The break-even analysis does not constitute evidence that any of these conditions can or will
be achieved. It quantifies the design space: how much each parameter would need to change,
independently, for $G_\mathrm{model}$ to cross zero. When multiple parameters improve
simultaneously, per-variable requirements decrease; a 20\% reduction in $\Delta I$ combined with
a 15\% reduction in $T^*$ would, for instance, bring $G_\mathrm{model}$ approximately to zero.
This is a design-space observation, not a prediction.

### 7.5 Implications for Cost-Aware Memory Management

The core methodological contribution is the separation of policy quality from economic
viability as independently falsifiable claims. Quality (whether the policy makes better
decisions) is evaluated by the fluid model at low cost; viability (whether improved decisions
pay for execution cost) requires a hardware measurement with a precisely defined timer boundary.
The two evaluations can — and in this case do — produce opposite verdicts.

This separation has practical implications for future evaluations of ML-based memory management
policies. The existing landscape (Table 1) consistently evaluates quality while omitting or
approximating the viability check. LLAMA addresses CPU overhead through engineering judgment;
RecMG argues that inference cost is hidden by pipelining. Both strategies may be correct in
their respective deployment contexts, but neither constitutes a formal falsifiable test of the
form $G_\mathrm{model} > 0$. The present evaluation demonstrates that stating this condition
explicitly and testing it pre-registered can produce a negative outcome that would not be
visible from quality metrics alone.

The specific failure mode — per-chunk feature computation and lookup cost exceeding per-chunk
decision value — is not necessarily universal. The economic outcome depends on the three-way
relationship between the exchange rate $c$, the per-decision value (determined by workload and
fluid model), and the implementation cost $\Delta I$. Different hardware environments, workload
characteristics, or policy implementations may yield different outcomes.

This study does not support the conclusion that ML-based memory reclamation is generally
unviable. It supports the more limited conclusion that the P1b-based C3 policy, evaluated on
BurstGPT trace-driven chunk pool simulation at T* = 2.6 s on the specified hardware, does not
satisfy $G_\mathrm{model} > 0$ under the pre-registered measurement protocol.

---

## 8. Limitations

The following limitations bound the scope of conclusions that can be drawn from this evaluation.

**L1. BurstGPT is a request-level serving trace, not a memory allocation log.**
The BurstGPT dataset \cite{burstgpt2025} records LLM serving requests (timestamp, session ID,
model, token counts) collected from Azure OpenAI. It does not record allocator-level events:
no chunk arrival or departure timestamps, no actual memory addresses, and no RSS measurements
are present in the original data. The chunk pool workload is constructed by mapping serving
request patterns to allocation and deallocation events through a preprocessing pipeline. The
resulting workload captures the temporal structure of LLM inference demand, but chunk arrival
and departure times are model-derived rather than directly observed. Conclusions about the
magnitude of memory management cost are conditioned on this workload construction.

**L2. The fluid model does not execute actual memory operations.**
The chunk pool fluid model tracks $R$ and $H$ as accounting quantities derived from the trace
and policy decisions, without issuing actual OS memory management calls (madvise, mmap, munmap).
The values of $\Delta R$, $\Delta H$, and G1\_SMODEL\_C are simulation-derived, not measurements
of physical memory state. The model is validated against a 160-window physical replay (maximum
deviation 0.2 MiB) for return volumes, but this validation does not extend to the full policy
cost distribution.

**L3. No live allocator experiment was conducted.**
This study does not evaluate C3 integrated into an actual jemalloc or TCMalloc allocator under
a live application workload. All policy decisions are made within the trace-driven simulator.
The incremental overhead $\Delta I$ is measured for the simulator's C++ implementation, not for
an allocator-integrated deployment. Whether allocator integration would increase or decrease
overhead relative to the simulator is not known.

**L4. No actual OS page reclamation was measured.**
The evaluation measures allocator-layer chunk return decisions, not kernel-layer page
reclamation. The actual reclamation path (MGLRU, 2QLRU, swap) and its overhead are outside
the measurement scope. The connection between chunk pool return decisions and realized RSS
reduction depends on OS reclamation latency and workload memory pressure, neither of which
is evaluated here.

**L5. No actual RSS or cloud billing savings were measured.**
$G_\mathrm{model}$ is a model-based economic estimate expressed in CPU-time units, not a
realized reduction in container resident set size or in cloud billing. The exchange rate $c$
and break-even time $T^*$ are measured on the evaluation platform and used to convert
fluid-model memory quantities to CPU-time equivalents. Whether a positive $G_\mathrm{model}$
would translate to measurable RSS reduction or billing savings in a production deployment is
not evaluated.

**L6. Single hardware platform.**
All CPU measurements are conducted on a single platform: Intel Core Ultra 7 155H, WSL2 Linux
kernel 6.6.87.2, g++ 13.3.0. The measured values of $c$ (281.8 µs/MiB), $T^*$ (2.6 s), and
$\Delta I$ (mean 2,405.932 ms) are specific to this environment. Memory return and reacquisition
costs differ across hardware: different cache hierarchies, DRAM latencies, and kernel
implementations will produce different $c$ and $T^*$ values. The economic outcome
$G_\mathrm{model}$ cannot be directly transferred to other hardware environments without
re-measuring these parameters.

**L7. Single-core execution.**
The G2W3 measurement pins execution to core 1 (taskset -c 1) to minimize NUMA effects and
cache interference. Production memory allocators may execute reclamation operations across
multiple cores or under concurrent application pressure. Single-core pinning eliminates sources
of timing variability at the cost of not reflecting multi-threaded allocator behavior. The
incremental overhead $\Delta I$ under concurrent execution is not evaluated.

**L8. Three official CPU measurement repetitions.**
The G2W3 protocol performs three official measurement iterations. The observed range of
$\Delta I$ across three iterations is 180.092 ms (minimum 2,329.017 ms, maximum 2,509.109 ms).
All three iterations produce $G_\mathrm{model} < 0$. The small number of repetitions means the
sample does not fully characterize the measurement distribution under different system states,
background processes, or OS scheduling behavior.

**L9. Population standard deviation from $n = 3$ is a descriptive statistic.**
The reported population standard deviation $\sigma_{\Delta I} = 75.831$ ms is a descriptive
summary of the three observed values, not an estimate of the true measurement population
variance. With $n = 3$, no meaningful confidence interval for the true mean $\Delta I$ can be
constructed. The finding that all three iterations fail $G_\mathrm{model} > 0$ by margins of
768–948 ms is reported as a descriptive outcome, not a statistical inference claim about the
probability of viability over the full measurement distribution.

**L10. Single policy variant.**
The economic evaluation applies exclusively to the P1b-based C3 policy as implemented in the
G2W3 C++ harness. Neither the 15-feature Cox GBDT used in the v1 AI path, the Python
scikit-learn implementation evaluated in Task A, nor any hypothetical reduced-overhead variant
is measured in G2. The break-even analysis (§6.9, §7.4) identifies directions for future policy
design but does not evaluate any specific alternative.

**L11. No sub-operation CPU breakdown.**
The G2W3 protocol measures aggregate $\Delta I$ in a single timer scope. It does not instrument
individual functions within the incremental path. The relative contributions of
compute\_p1b\_features, compute\_Tk, online\_kappa\_T, and the two run\_fallback3 calls to
the total $\Delta I$ are not known from the present measurement. Any attribution of overhead
to individual operations would be speculative.

**L12. No cross-environment generalization.**
The specific combination of T* = 2.6 s, $c = 281.8$ µs/MiB, $\Delta I = 2{,}405.932$ ms, and
the BurstGPT workload structure is unique to this evaluation environment. The economic outcome
may differ qualitatively under different memory latency profiles (affecting $c$), different
container resource limits (affecting $T^*$), different request patterns (affecting $\Delta R$
and $\Delta H$), or different hardware ISA characteristics. This study provides no evidence that
the G2 FAIL result generalizes beyond the described evaluation scope.

---

## 9. Conclusion

### 9.1 Research Question

This paper addresses the following question: *Can predictive memory reclamation pay for its own
CPU cost?* In a containerized environment where CPU time is a billed, quota-limited resource, an
AI-augmented return policy is economically justified only if the CPU-equivalent value of its
improved memory decisions exceeds the CPU cost of making those decisions.

### 9.2 Cost Model and Evaluation Method

We instantiate this question through a formal cost model $C = R + H/T^*$ that expresses memory
management cost in CPU-time units using a measured exchange rate $c = 281.8$ µs/MiB and
break-even holding time $T^* = 2.6$ s. The net gain metric
$G_\mathrm{model} = c(\Delta R + \Delta H/T^*) - \Delta I$ provides an explicit pre-registered
economic viability condition ($G_\mathrm{model} > 0$), tested on a C++ implementation with a
precisely defined timer boundary (G2W3: CLOCK\_PROCESS\_CPUTIME\_ID, single core).

The evaluation proceeds in two independent pre-registered phases. Phase G1 estimates the
simulation-based memory savings value G1\_SMODEL\_C through fluid-model simulation (no actual
memory operations, no OS system calls). Phase G2 measures the incremental CPU overhead $\Delta I$
directly using a C++ harness, with computational equivalence verified by Python–C++ parity
checking (420/420 windows PASS, zero deviation).

### 9.3 Policy Quality Results

The P1b-based C3 policy, using a 65$^3$ distilled lookup table derived from a 15-feature Cox
GBDT, produces positive memory management quality in the fluid model. All three v2 sealed
hypotheses are confirmed at T* = 2.6 s:

- H1v2: C3 − rule = −2.34 %p (threshold < −0.5 %p), **PASS**
- H2v2: C3 − fixed-online = −9.02 %p (threshold < −0.5 %p; retreat ≤ 10\%), **PASS**
- H3v2: synthetic schedules 27/9/4, $p = 3.4 \times 10^{-5}$ (Holm < 0.05), **PASS**

The placebo analysis confirms that 99.4–99.8\% of C3's simulation gain originates from
per-chunk prediction correspondence, not switching structure or aggregate statistics.

The simulation-based economic value of these decisions decomposes as:
$c \cdot \Delta R = -255.311$ ms (from $\Delta R = -906.0$ MiB; C3 returns more than rule) and
$c \cdot \Delta H/T^* = +1{,}816.171$ ms (from $\Delta H = +16{,}756.72$ MiB·s; C3 reduces
idle holding), yielding G1\_SMODEL\_C = $+1{,}560.860$ ms.

### 9.4 Economic Viability Results

**Table 14.** G2W3 primary outcome summary.

| Metric | Value |
|--------|-------|
| Mean $\Delta I$ (3 repetitions, meas) | 2,405.932 ms |
| Population SD $\sigma_{\Delta I}$ | 75.831 ms |
| G1\_SMODEL\_C (sim) | 1,560.860 ms |
| Mean $G_\mathrm{model}$ (calc) | **−845.072 ms** |
| Mean $G_\mathrm{CPU}$ (calc) | −2,661.243 ms |
| $G_\mathrm{model} > 0$ satisfied? | **No** (all 3 iterations fail) |

All three measurement repetitions independently fail the pre-registered viability criterion,
with $G_\mathrm{model}$ ranging from $-768.157$ ms (iteration 1) to $-948.249$ ms (iteration 3).
The range across iterations (180.092 ms) is substantially smaller than the mean shortfall
(845.072 ms), ruling out measurement variability as an explanation for the negative outcome.

### 9.5 Answer to the Research Question

Under the conditions of this evaluation, **no** — the P1b-based C3 policy does not recoup its
CPU cost from memory savings. The incremental execution overhead $\Delta I = 2{,}405.932$ ms
exceeds the simulation-based memory savings value G1\_SMODEL\_C = 1,560.860 ms by 845.072 ms,
and $G_\mathrm{model} = -845.072$ ms in the mean across all three official measurement
repetitions.

Break-even sensitivity analysis (§6.9) indicates that economic viability would require, in
single-variable terms: a 35.1\% reduction in $\Delta I$, a 31.8\% reduction in $T^*$, a 54.1\%
increase in $c$, a 46.5\% increase in $\Delta H$, or a sign-reversal-plus-increase in $\Delta R$
of approximately 2,999 MiB. None of these conditions is currently satisfied.

### 9.6 Scope of Contributions

The conclusions of this study are bounded by the evaluation scope. The G2 FAIL result applies
specifically to: the P1b-based C3 policy, evaluated under BurstGPT trace-driven chunk pool
simulation, at T* = 2.6 s, on a single-core Intel Core Ultra 7 155H / WSL2 platform. This
result does not generalize to other ML-based memory management policies, other hardware
environments, other workloads, other implementations of the same policy, or fluid model
configurations with different $T^*$ or $c$ values.

The methodological contributions — the two-component cost model with measured exchange rate,
the G2W3 measurement boundary, the Python–C++ parity protocol, and the pre-registered
break-even condition — are potentially applicable to future evaluations of ML memory management
policies regardless of the direction of the result.

### 9.7 Future Directions

The break-even analysis identifies three directions for future investigation:

1. **Reducing $\Delta I$**: Restructuring the C3 incremental path — specifically the five
   operations that constitute all of $\Delta I = 2{,}405.932$ ms — to reduce feature computation
   and lookup overhead. Their individual CPU shares are not measured in this evaluation and would
   need to be characterized first.

2. **Deployment contexts with different $c$ or $T^*$**: The economic exchange rate is hardware-
   and deployment-specific. Environments with higher memory latency (larger $c$) or stricter
   container memory limits (smaller effective $T^*$) might satisfy $G_\mathrm{model} > 0$ for
   the same policy implementation.

3. **Policy variants that increase $\Delta H$ while reducing $\Delta I$**: Policies with
   improved timing predictions could increase G1\_SMODEL\_C; if combined with lower overhead,
   the viability threshold may be crossed.

Each direction requires a new pre-registered evaluation with an independently measured $\Delta I$.
Retroactive adjustment of parameters to satisfy $G_\mathrm{model} > 0$ post-measurement is not
valid under the pre-registration framework.

---

## Verification Checklist

| Check | Status |
|---|---|
| 1. All formula signs and units consistent throughout | ✓ — ΔR = R\_rule − R\_C3 (neg when C3 returns more); ΔH = H\_rule − H\_C3 (pos when C3 holds less); ΔI = CPU\_C3 − CPU\_rule (pos) |
| 2. ΔH definition consistent across document | ✓ — H\_rule − H\_C3 throughout §3–§9 |
| 3. G1 and G2 purposes distinguished | ✓ — §3.4, §4.6, §6.7, §6.8, §7.1, §9.2 each explain the distinction |
| 4. Official G2 results match measure-summary-official.json | ✓ — values copied verbatim from raw.json and measure-summary-official.json |
| 5. Evaluation scope not overstated | ✓ — fluid model / simulation vs. meas. / calc labeled throughout; §8 L1–L5 explicitly bound scope |
| 6. §7 term decomposition consistent with §3 and §6 | ✓ — Table 13 values: −255.311 + 1816.171 = 1560.860; 1560.860 − 2405.932 = −845.072 |
| 7. §8 limitations do not describe experiments not performed | ✓ — L3 (no live allocator), L4 (no OS reclamation), L5 (no RSS measurement) each state what was NOT done |
| 8. §9 conclusion does not generalize G2 FAIL beyond evaluation scope | ✓ — §9.5–§9.6 state result applies to P1b-based C3 at T* = 2.6 s on specified hardware only |
| 9. All citation keys match BibTeX | ✓ — 11 keys: llama2020, tmo2022, memtis2023, lecar2018, memperf2023, recmg2025, kml2023, jemalloc, tcmalloc, burstgpt2025, pagereclaim2025 |
| 10. No values added beyond verified source files | ✓ — all numbers traceable to final-report-draft.md, measure-summary-official.json, raw.json, g1/simulation/summary.json |

---

## Manuscript Status

**Completed**: §1 Introduction, §2 Background and Related Work, §3 Problem Formulation,
§4 Methodology, §5 Experimental Setup, §6 Results, §7 Discussion, §8 Limitations, §9 Conclusion.

**G2 primary result** (from `measure-summary-official.json`, frozen SHA-256 `669587f1...`):
- Mean $\Delta I$ = 2,405.932 ms; $\sigma$ (pop.) = 75.831 ms
- Mean $G_\mathrm{model}$ = −845.072 ms; Mean $G_\mathrm{CPU}$ = −2,661.243 ms
- G2 pass: false (all three repetitions fail $G_\mathrm{model} > 0$)

---

*Git commit/push not performed. Experimental artifacts not modified.*
