# Can Predictive Memory Reclamation Pay for Its Own CPU Cost?

**A Pre-registered Evaluation of a Distillation-Based Chunk Pool Policy**

> Status: v1 draft — §1 Introduction, §2 Background and Related Work, §3 Problem Formulation.  
> All citations reference verified BibTeX keys in `docs/related-work-matrix.md` §8.  
> Do not interpret numbers beyond the stated evaluation scope (P1b-based C3, BurstGPT workload, WSL2 hardware, T* = 2.6 s).

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
reducing $\Delta I$ by at least 35.1\% (from 2,405.932 ms to below 1,560.860 ms) or reducing
$T^*$ by at least 31.8\% (from 2.6 s to below 1.774 s).

This paper makes the following contributions:

1. **Cost model and break-even criterion.** A two-component cost function $C = R + H/T^*$
   expressing memory management cost in CPU-time units, and a net gain metric
   $G_\mathrm{model} = c(\Delta R + \Delta H / T^*) - \Delta I$ with an explicit economic
   viability condition ($G_\mathrm{model} > 0$).

2. **Separation of prediction quality and execution cost.** A measurement design that evaluates
   the memory savings value (G1, simulation-based) and the CPU overhead (G2, hardware-measured)
   as independent, pre-registered phases, exposing the possibility that memory value is positive
   while net economic gain is negative.

3. **Complete measurement boundary.** A G2W3 timer scope that includes feature computation
   (compute\_p1b\_features), table lookup (compute\_Tk), online scaling (online\_kappa\_T), and
   both policy paths within a single process-CPU timer, ensuring no overhead component escapes
   the measurement.

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

Table 1 summarizes the key dimensions of related work alongside this study.

| Work | ML? | CPU overhead reported | Formal break-even? | Memory return domain |
|------|-----|-----------------------|--------------------|----------------------|
| jemalloc \cite{jemalloc} | No | No | No | Chunk pool (decay) |
| TCMalloc \cite{tcmalloc} | No | No | No | Chunk pool (rate) |
| LLAMA \cite{llama2020} | LSTM (offline) | 400 ns/call | No | Heap allocation |
| TMO \cite{tmo2022} | No (PSI heuristic) | Not reported | No | Fleet DRAM |
| MEMTIS \cite{memtis2023} | No (histogram) | \textless3\% (constraint) | No | Tiered memory |
| LeCaR \cite{lecar2018} | Online RL | Space only | No | Cache eviction |
| RecMG \cite{recmg2025} | LSTM+Attn | 92 µs/call | No | Tiered memory |
| KML \cite{kml2023} | Kernel-inline ML | \textless0.2\% | No | I/O scheduling |
| pagereclaim2025 \cite{pagereclaim2025} | No | Per-phase cycles | No | Kernel page reclaim |
| **This work** | P1b distilled table | ΔI = 2,405.932 ms | **Yes (G\_model > 0)** | Chunk pool return |

This work formalizes the question that existing work leaves implicit: does the CPU-equivalent
value of a predictive memory return policy's memory savings exceed the CPU cost of obtaining
those savings? We answer this question with a pre-registered protocol and report a negative
result, together with a sensitivity analysis that quantifies the conditions for future policy
variants to achieve economic viability.

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
empirically as the sum of the per-MiB return cost and the per-MiB reacquisition cost on the
evaluation platform:

$$c = c_\mathrm{return} + c_\mathrm{reacq} = 25.6 + 256.2 = 281.8 \ \mu\mathrm{s/MiB}$$

measured under WSL2 (Intel Core Ultra 7 155H, 1 MiB chunks, MADV\_DONTNEED, 300 repetitions per
trial, 3 trials). The break-even holding time is then determined by setting the
CPU cost of holding one MiB for $T^*$ equal to the reacquisition cost:

$$T^* = \frac{c_\mathrm{reacq}}{c / T^*} \approx 2.6 \ \mathrm{s}$$

(median of three measurements: 2.61, 2.64, 2.63 s; reported to two significant figures).

$C$ is defined within the fluid model of the chunk pool: $R$ and $H$ are computed from the
allocator's return decisions and the chunk arrival/departure trace, without executing actual
system calls on live memory. This model accurately tracks the return volume and idle holding
duration under a given policy, but does not account for the CPU time spent making the return
decision. The incremental decision cost is captured separately in the net gain metric below.

### 3.2 Net Economic Gain

We evaluate whether policy C3 is economically viable relative to the rule-based baseline through
the net gain metric:

$$G_\mathrm{model} = c \cdot \left(\Delta R + \frac{\Delta H}{T^*}\right) - \Delta I$$

where the quantities are defined as follows.

**$\Delta R = R_\mathrm{rule} - R_\mathrm{C3}$** (MiB): the difference in returned memory
between the rule policy and C3. A negative value indicates that C3 returns more memory than the
rule (R\_C3 > R\_rule). This term contributes negatively to $G_\mathrm{model}$ when C3 is more
aggressive in returning memory, reflecting higher future reacquisition exposure.

**$\Delta H = H_\mathrm{rule} - H_\mathrm{C3}$** (MiB·s): the reduction in idle holding cost
achieved by C3 compared to the rule. A positive value indicates that the rule policy accumulates
more idle holding cost than C3 (H\_rule > H\_C3), which counts in C3's favor. This term
contributes positively to $G_\mathrm{model}$ when C3 avoids prolonged idle retention.

**$\Delta I = \mathrm{CPU}_{C3} - \mathrm{CPU}_\mathrm{rule}$** (ms): the incremental CPU time
consumed by the C3 decision path relative to the rule path, measured by process-level CPU
timers within the G2W3 boundary (§4). This term is always non-negative in our measurements and
is the primary economic cost of using C3.

The term $c \cdot (\Delta R + \Delta H / T^*)$ — hereafter G1\_SMODEL\_C — is the
CPU-equivalent value of C3's memory management savings over rule, estimated from the fluid model.
$G_\mathrm{model} > 0$ holds if and only if this memory savings value strictly exceeds the
incremental CPU cost $\Delta I$. We pre-register $G_\mathrm{model} > 0$ for all measurement
repetitions as the economic viability criterion (G2 pass condition).

$G_\mathrm{model}$ is a model-based economic estimate, not a realized CPU reduction or cloud
billing saving. Its validity is bounded by the fluid-model approximation for $R$ and $H$, the
representativeness of $c$ and $T^*$ measured on the evaluation platform, and the
representativeness of the BurstGPT workload \cite{burstgpt2025} for the target use case.

For analytical decomposition, we also report the CPU-only net gain:

$$G_\mathrm{CPU} = c \cdot \Delta R - \Delta I$$

which excludes the holding-cost term $\Delta H / T^*$ and isolates the question of whether C3's
incremental return volume alone justifies its CPU cost. Because the fluid model imposes no memory
capacity constraint, $G_\mathrm{CPU}$ cannot serve as the primary viability criterion: a policy
that never returns memory would eliminate the negative contribution of $\Delta R$ while
accumulating unbounded idle holding cost. We use $G_\mathrm{CPU}$ as a decomposition aid to
characterize the relative contributions of the return and holding components.

### 3.3 Sign Conventions and Current Values

Under the P1b-based C3 policy, the fluid model over 420 BurstGPT evaluation windows yields:

| Variable | Value | Interpretation |
|---|---|---|
| $\Delta R = R_\mathrm{rule} - R_\mathrm{C3}$ | −906.0 MiB | C3 returns more memory than rule (R\_C3 > R\_rule) |
| $\Delta H = H_\mathrm{rule} - H_\mathrm{C3}$ | 16,756.72 MiB·s | Rule accumulates more idle holding than C3 |
| G1\_SMODEL\_C = $c(\Delta R + \Delta H/T^*)$ | 1,560.860 ms | Net memory benefit, CPU-equivalent |
| $\Delta I$ (mean, 3 repetitions) | 2,405.932 ms | Incremental CPU cost of C3 path |
| $G_\mathrm{model}$ (mean) | **−845.072 ms** | Net economic gain: **negative** |

The negative $\Delta R$ requires careful interpretation. It does not indicate that C3 performs
poorly in the memory dimension: C3 is *more* aggressive in returning memory than the rule policy.
The penalty from $c \cdot \Delta R < 0$ reflects that this additional returned memory may incur
reacquisition cost $c$ per MiB. The positive $\Delta H$ captures the compensating benefit: by
returning memory sooner, C3 avoids prolonged idle holding. In the present experiment, the
holding-cost savings ($c \cdot \Delta H/T^* \approx 1{,}816.1$ ms) outweigh the reacquisition
exposure ($c \cdot |\Delta R| \approx 255.3$ ms), yielding a positive G1\_SMODEL\_C.
However, this combined savings (1,560.860 ms) is insufficient to cover the measured $\Delta I$
(2,405.932 ms), producing $G_\mathrm{model} = -845.072$ ms.

The break-even sensitivity analysis in §6 examines the magnitude of change required in each
variable to bring $G_\mathrm{model}$ to zero. The key conditions are: $\Delta I < 1{,}560.860$
ms (a reduction of 35.1\% from the current mean); $T^* < 1.774$ s (a reduction of 31.8\%); $c >
434.371$ µs/MiB (an increase of 54.1\%); $\Delta H > 24{,}553.699$ MiB·s (an increase of
46.5\%); or $\Delta R > +2{,}092.837$ MiB (a sign reversal of approximately +2,999 MiB,
corresponding to C3 returning *less* memory than rule — the opposite of its current behavior).

### 3.4 Two Evaluation Dimensions

The evaluation proceeds in two distinct phases that test complementary hypotheses.

**Phase G1 (simulation-based value estimation).** We estimate G1\_SMODEL\_C = $c(\Delta R +
\Delta H/T^*)$ by running both C3 and rule over the fluid model of 420 BurstGPT evaluation
windows. This phase uses the P1b distilled lookup table (3 features: height, usage level, stack
level; 65³ grid) and determines whether C3 creates positive memory management value over rule.
G1 is an exploratory, pre-registered selection phase: its purpose is to confirm that the memory
savings value is large enough to warrant CPU measurement (G1 pass criterion: G1\_SMODEL\_C ≥ 1.0
s), not to certify economic viability. The G1 result, G1\_SMODEL\_C = 1,560.860 ms, satisfies
this criterion.

**Phase G2 (CPU overhead measurement).** We measure $\Delta I$ directly using a C++ harness
with process-level CPU timers across 420 evaluation windows and 3 measurement repetitions (G2W3
format). The G2 pass criterion — $G_\mathrm{model} > 0$ for all three repetitions — is
pre-registered and was committed before the measurement was executed. The G2 result is the
primary finding of this paper: all three repetitions fail the criterion
($G_\mathrm{model} \in \{-768.157, -818.811, -948.249\}$ ms).

The relationship between G1 and G2 is hierarchical but not equivalent. A positive G1 outcome
(G1\_SMODEL\_C > 0) is a necessary but not sufficient condition for a positive G2 outcome
($G_\mathrm{model} > 0$): the memory savings value must exist before it can cover $\Delta I$,
but its existence does not guarantee that it is large enough. In this experiment, the G1
hypothesis is confirmed and the G2 hypothesis is rejected. The gap between G1\_SMODEL\_C
(1,560.860 ms) and $\Delta I$ (2,405.932 ms average) defines the economic shortfall (845.072 ms)
that future policy variants must close.

---

## Self-Review

| Check | Status |
|---|---|
| 1. Research question stated in Introduction | ✓ — explicit quote in §1 ¶5 |
| 2. Differentiation from existing work backed by comparison | ✓ — Table 1; RecMG/LLAMA compared in §2.4/§2.2 |
| 3. RecMG comparison fair (contributions not minimized) | ✓ — 43% speedup stated; pipelining argument acknowledged as valid in its setting |
| 4. G\_model and G\_CPU defined consistently | ✓ — §3.2; sign conventions explicit in §3.3 |
| 5. ΔR sign interpretation correct | ✓ — §3.3 "C3 returns more memory than rule (R\_C3 > R\_rule)" |
| 6. All citation keys exist in BibTeX | ✓ — 11 keys: llama2020, tmo2022, memtis2023, lecar2018, memperf2023, recmg2025, kml2023, jemalloc, tcmalloc, burstgpt2025, pagereclaim2025 |
| 7. No unsupported claims or exaggerated novelty | ✓ — "to our knowledge" used for novelty claims; no "first" assertions |
| 8. Three sections logically connected | ✓ — §1 → §2 (gap identification) → §3 (formal problem statement) |

---

## Manuscript Status

**Sections written**: §1 Introduction, §2 Background and Related Work, §3 Problem Formulation.  
**Sections pending**: §4 Methodology, §5 Experimental Setup, §6 Results, §7 Discussion,
§8 Limitations, §9 Conclusion.

**Citations used**: 11 (all verified BibTeX keys — llama2020, tmo2022, memtis2023, lecar2018,
memperf2023, recmg2025, kml2023, jemalloc, tcmalloc, burstgpt2025, pagereclaim2025).

**Unconfirmed items**:
- RecMG 92 µs inference figure and "pipelining" claim are cited from §2.6 of `related-work-matrix.md`; the HPCA 2025 paper body was not directly read. These claims are attributed to RecMG with standard academic hedging.
- LeCaR hit-rate evaluation and space overhead report are based on USENIX abstract-level information; the HotStorage 2018 full paper was not read.
- KML <0.2% overhead figure is from the TOCS abstract-level entry in `related-work-matrix.md`.

**Additional evidence needed before subsequent sections**:
- §4 Methodology: source-level G2W3 timer boundary (available in `exp/v2dev/g2/g2_harness.cpp`).
- §5 Experimental Setup: hardware specs and software version table (available in `docs/final-report-draft.md` §7.2).
- §6 Results: pre-registered result files (`docs/sealed-results-v1.md`, `docs/sealed-results-v2.md`, `exp/v2dev/g2/measurement/measure-summary-official.json`).

---

*Git commit/push not performed. Experimental artifacts not modified.*
