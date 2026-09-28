// Generated from the Rust types by mb-server (src/types.rs). Do not edit:
// run `UPDATE_TYPES=1 cargo test -p mb-server --test types` after changing them.

export type ArchSummary = { family: string | null, architectures: Array<string>, num_layers: number, hidden_size: number | null, vocab_size: number | null, max_positions: number | null, tie_word_embeddings: boolean | null, rope_theta: number | null, rope_scaling: JsonValue | null, layer_pattern: LayerPattern, mtp_modules: number, multimodal: boolean, };

export type AttentionKind = "mha" | "gqa" | "mqa" | "mla";

export type AttentionSpec = { kind: AttentionKind, num_heads: number | null, num_kv_heads: number | null, head_dim: number | null, 
/**
 * Sigmoid output gate on attention (Qwen3-Next style).
 */
output_gate: boolean, qk_norm: boolean, 
/**
 * Sliding-window size, if this layer attends locally.
 */
sliding_window: number | null, mla: MlaSpec | null, };

export type Backend = "cuda" | "metal";

export type Catalog = { features: Array<CatalogFeature>, hardware: Array<HardwareProfile>, };

export type CatalogFeature = { id: string, title: string, summary: string, };

export type Compat = { 
/**
 * Reasons the feature can't be applied as requested.
 */
blockers: Array<string>, 
/**
 * Applicable, but with caveats.
 */
warnings: Array<string>, };

export type Component = "embedding" | "lm_head" | "final_norm" | "layer" | "mtp" | "multimodal" | "other";

export type ComputeEstimate = { profile: string, train_flops: Range, gpu_hours: Range, wall_hours: Range, 
/**
 * Peak memory per device with the frozen trunk held in BF16.
 */
peak_gib_per_gpu: number, 
/**
 * Same, if the frozen trunk stays packed at the source bit width.
 */
peak_gib_per_gpu_packed_trunk: number, 
/**
 * GPU-hours to compute precomputed features (one trunk forward per token),
 * separate from training; `None` when no stage precomputes.
 */
extraction_gpu_hours: Range | null, fits: Fit, notes: Array<string>, };

export type Confidence = "low" | "medium" | "high";

/**
 * How much of a job a client has seen.
 */
export type Cursor = { events: number, log: number, 
/**
 * Bumped on every change to the job, including its status.
 */
version: number, };

/**
 * Storage type of a tensor as it sits on disk.
 *
 * Plain element types come from safetensors (and GGUF's unquantized types);
 * block-quantized GGUF types are carried as [`DType::Ggml`] so unknown or
 * vendor-specific types (e.g. PrismML's `PQ2_0`) survive a round trip.
 */
export type DType = "bool" | "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64" | "i64" | "f8_e4_m3" | "f8_e5_m2" | "f8_e8_m0" | "f16" | "bf16" | "f32" | "f64" | { "ggml": GgmlType };

export type DTypeStat = { dtype: string, tensors: number, params: number, bytes: number, 
/**
 * False if any tensor's size had to be inferred (unknown GGUF type).
 */
bytes_exact: boolean, 
/**
 * Measured storage bits per parameter for this dtype.
 */
bits_per_param: number, };

export type Detection = { "state": "absent" } | { "state": "present", "detail": string } | { "state": "partial", "detail": string };

export type DirEntry = { name: string, path: string, kind: EntryKind, 
/**
 * File size in bytes (files only).
 */
size: number | null, };

export type DirListing = { path: string, parent: string | null, 
/**
 * Directories first, then files, each sorted by name. Hidden entries are skipped.
 */
entries: Array<DirEntry>, };

/**
 * A measurable runtime effect of the feature, e.g. KV bytes per token.
 */
export type Effect = { metric: string, before: number, after: number, unit: string, };

export type EntryKind = "dir" | "gguf" | "hf_model" | "recipe" | "job_spec" | "events" | "other";

export type ErrorBody = { error: string, };

/**
 * A plugin's estimate before pricing on hardware.
 */
export type Estimate = { effects: Array<Effect>, stages: Array<Stage>, risk: QualityRisk, assumptions: Array<string>, confidence: Confidence, references: Array<string>, };

/**
 * An event with its envelope fields.
 */
export type EventRecord = { schema_version: number, time: number, stage: string | null, } & ({ "event": "started", job_id: string, device: string, trainable_params: number | null, train_tokens: number | null, eval_tokens: number | null, } | { "event": "progress", step: number, steps: number, tokens: number | null, loss: number, accuracy: number | null, lr: number | null, tokens_per_s: number | null, } | { "event": "eval", step: number, loss: number, accuracy: number, tokens: number | null, } | { "event": "checkpoint", step: number, path: string, } | { "event": "finished", status: string, outputs: { [key in string]: string }, } | { "event": "error", message: string, });

export type FeaturePlan = { id: string, title: string, summary: string, params: { [key in string]: JsonValue }, detection: Detection, compat: Compat, 
/**
 * Absent when compatibility checks block the feature.
 */
estimate: Estimate | null, compute: Array<ComputeEstimate>, surgery: Array<string>, export_notes: Array<string>, };

export type FfnSpec = { "type": "dense", intermediate_size: number | null, } | { "type": "moe" } & MoeSpec | { "type": "none" };

export type Fit = "not_applicable" | "yes" | "with_packed_trunk" | "no";

/**
 * A GGML tensor type id. Ids outside the known table are kept verbatim.
 */
export type GgmlType = number;

export type HardwareProfile = { id: string, description: string, backend: Backend, gpus: number, 
/**
 * Memory per device, GiB (unified memory for Apple Silicon).
 */
mem_per_gpu_gib: number, 
/**
 * Share of device memory a training job can actually use.
 */
usable_fraction: number, 
/**
 * Spec-sheet dense BF16/FP16 TFLOPS per device.
 */
peak_tflops: number, };

export type Health = { name: string, version: string, 
/**
 * Whether the web UI build is being served.
 */
web_ui: boolean, };

export type InspectRequest = { path: string, 
/**
 * Context length for KV-cache totals (defaults to the model's max positions).
 */
context: number | null, 
/**
 * Also return the tensor index.
 */
tensors: boolean, };

export type InspectResponse = { report: Report, 
/**
 * Per-layer structure, in order.
 */
layers: Array<LayerRow>, 
/**
 * The tensor index, when requested.
 */
tensors: Array<TensorRow> | null, };

export type JobSource = { "kind": "spec", spec_path: string, 
/**
 * Python interpreter with `modelbuilder_train` installed (default: the manager's).
 */
python: string | null, } | { "kind": "events", events_path: string, };

export type JobStatus = "running" | "succeeded" | "failed" | "cancelled";

export type JobSummary = { id: number, source: JobSource, status: JobStatus, 
/**
 * The spec's `job_id` (or the `started` event's, when following a file).
 */
job_id: string | null, 
/**
 * Unix seconds.
 */
started_at: number, finished_at: number | null, events: number, latest_progress: EventRecord | null, latest_eval: EventRecord | null, outputs: { [key in string]: string }, error: string | null, };

/**
 * What changed since a [`Cursor`].
 */
export type JobUpdate = { summary: JobSummary, 
/**
 * Event records from `since.events` on.
 */
events: Array<EventRecord>, 
/**
 * Log lines (stderr and non-event stdout) from `since.log` on.
 */
log: Array<string>, 
/**
 * Pass this back to get the next update.
 */
cursor: Cursor, };

export type JsonValue = number | string | boolean | Array<JsonValue> | { [key in string]: JsonValue } | null;

/**
 * Estimated per-sequence inference memory for attention state.
 */
export type KvCacheEstimate = { 
/**
 * Attention layers whose cache grows with the sequence.
 */
global_layers: number, 
/**
 * Attention layers bounded by a sliding window.
 */
windowed_layers: number, 
/**
 * Linear-attention / SSM layers (fixed-size state, no per-token cache).
 */
linear_layers: number, 
/**
 * Cached elements per token across global layers.
 */
elements_per_token: number, context: number | null, precisions: Array<KvPrecision>, 
/**
 * Fixed recurrent state of linear layers per sequence, in bytes (fp32 state, bf16 conv).
 */
linear_state_bytes: number | null, 
/**
 * Layers skipped because their dimensions are unknown.
 */
unknown_layers: Array<number>, assumptions: Array<string>, };

/**
 * KV cache storage precisions to report. Bits per element include scales.
 */
export type KvPrecision = { name: string, bits_per_element: number, 
/**
 * KV bytes per token across all unbounded (global) attention layers.
 */
bytes_per_token: number, 
/**
 * Total KV bytes for one sequence at `context` tokens, including sliding-window layers.
 */
bytes_at_context: number, };

/**
 * Singular-value summary of one attention layer's K and V projections.
 */
export type KvSpectrum = { layer: number, 
/**
 * Rows of K and of V (`kv_heads × head_dim` each), and the input width.
 */
k_rows: number, v_rows: number, cols: number, k: SpectrumSummary, v: SpectrumSummary, 
/**
 * `[K; V]` stacked: the rank a shared KV latent (MLA-style) would need.
 */
kv: SpectrumSummary, };

export type Layer = { index: number, mixer: Mixer, ffn: FfnSpec, params: number, };

/**
 * Run-length encoded layer layout: `repeats × [count × label, ...]`, plus a prefix
 * of leading layers that don't fit the repeating block (e.g. DeepSeek's dense first layers).
 */
export type LayerPattern = { prefix: Array<[number, string]>, repeats: number, block: Array<[number, string]>, };

export type LayerRow = { layer: Layer, 
/**
 * Short label, e.g. `gqa(24q/4kv,d256)+dense` (the CLI's layer pattern notation).
 */
label: string, };

export type LinearAttentionSpec = { 
/**
 * e.g. `gated_deltanet`, `mamba`, `rwkv`, or `unknown`.
 */
variant: string, num_key_heads: number | null, num_value_heads: number | null, key_head_dim: number | null, value_head_dim: number | null, conv_kernel: number | null, };

export type ListRequest = { 
/**
 * Directory to list; the current directory if empty.
 */
path: string, };

/**
 * The token-mixing block of a layer.
 */
export type Mixer = { "type": "attention" } & AttentionSpec | { "type": "linear_attention" } & LinearAttentionSpec | { "type": "unknown" };

export type MlaSpec = { kv_lora_rank: number | null, q_lora_rank: number | null, qk_rope_head_dim: number | null, qk_nope_head_dim: number | null, v_head_dim: number | null, };

export type MoeSpec = { num_experts: number | null, experts_per_token: number | null, num_shared_experts: number, expert_intermediate_size: number | null, };

/**
 * Parameter counts by role. Counts are logical elements, so block-quantized
 * GGUF tensors count their true weights; packed safetensors (e.g. int4 in U8)
 * count storage elements; see `QuantSummary` for bytes.
 */
export type ParamBreakdown = { total: number, embedding: number, lm_head: number, attention: number, linear_attention: number, ffn_dense: number, moe_routed_experts: number, moe_shared_experts: number, router: number, norms: number, mtp: number, multimodal: number, other: number, 
/**
 * Trunk parameters used per token (routed experts scaled by top-k / experts).
 */
active_per_token: number, };

export type Plan = { model: string, hardware: Array<HardwareProfile>, features: Array<FeaturePlan>, schedule: Schedule, cost_model_assumptions: Array<string>, };

export type PlanRequest = { 
/**
 * The model; overrides the recipe's source.
 */
path: string | null, 
/**
 * Recipe TOML text (the same format as `plan --recipe`).
 */
recipe: string | null, 
/**
 * Feature specs, `id` or `id:key=value,...`, added to the recipe's.
 * With no features at all, the whole catalog is evaluated.
 */
features: Array<string>, 
/**
 * Hardware profile ids; override the recipe's. Empty: all profiles.
 */
hardware: Array<string>, };

/**
 * What the checkpoint says about where it came from and how it was trained.
 */
export type Provenance = { name: string | null, license: string | null, base_models: Array<string>, tags: Array<string>, has_chat_template: boolean, 
/**
 * Declared training dtype (`torch_dtype`/`dtype`).
 */
dtype: string | null, 
/**
 * Context the model was pre-trained at, if RoPE scaling reveals it.
 */
original_context: number | null, 
/**
 * Context the checkpoint is configured for.
 */
configured_context: number | null, 
/**
 * Best guess at base vs. post-trained, with the reason.
 */
lineage_hint: string, };

export type QualityRisk = { level: RiskLevel, expected: string, recovery: string, };

export type QuantSummary = { by_dtype: Array<DTypeStat>, total_bytes: number, 
/**
 * Storage bits per parameter across the whole checkpoint.
 */
bits_per_param: number, 
/**
 * Same, for the main trunk's layer weights only (the part that's usually quantized).
 */
trunk_bits_per_param: number, 
/**
 * What the checkpoint declares (HF `quantization_config`, GGUF `general.file_type`).
 */
declared: JsonValue | null, 
/**
 * Rotation folded into the stored weights, if the checkpoint declares one.
 */
rotation: WeightRotation | null, notes: Array<string>, };

export type Range = { low: number, high: number, };

export type Report = { source: SourceSummary, architecture: ArchSummary, params: ParamBreakdown, quantization: QuantSummary, kv_cache: KvCacheEstimate, provenance: Provenance, warnings: Array<string>, };

export type RiskLevel = "low" | "medium" | "high";

/**
 * All compatible features' stages in dependency order, with totals per profile.
 */
export type Schedule = { stages: Array<ScheduledStage>, 
/**
 * The whole schedule priced per hardware profile (the peak is the largest stage's).
 */
totals: Array<ComputeEstimate>, notes: Array<string>, };

/**
 * One stage in the order the whole plan should run.
 */
export type ScheduledStage = { 
/**
 * 1-based position.
 */
order: number, feature: string, stage: string, trunk: TrunkUse, 
/**
 * Why it runs at this position.
 */
reason: string, };

export type SignMode = "identity" | "explicit";

export type SourceFormat = "hf_safetensors" | "gguf";

export type SourceSummary = { format: SourceFormat, path: string, files: number, tensors: number, };

export type SpectrumSummary = { 
/**
 * Smallest rank capturing 90%, 95% and 99% of the squared singular values.
 */
energy_rank_90: number, energy_rank_95: number, energy_rank_99: number, 
/**
 * exp(entropy) of the normalized singular values (Roy & Vetterli).
 */
effective_rank: number, full_rank: number, };

/**
 * One training stage, described in terms the cost model can price.
 */
export type Stage = { name: string, what: string, 
/**
 * Parameters updated by the optimizer.
 */
trainable_params: number, 
/**
 * Parameters that activation gradients flow back through (at least
 * `trainable_params`; everything above the lowest trainable layer).
 */
backprop_params: number, 
/**
 * Training tokens.
 */
tokens: Range, seq_len: number, loss: string, data: string, 
/**
 * Whether a teacher forward pass (the unmodified model) runs per token.
 */
teacher_forward: boolean, trunk: TrunkUse, 
/**
 * For frozen stages: the trunk's outputs are computed once beforehand
 * (in the serving runtime) and stored, so training never runs or holds
 * the trunk. Only the embedding and the tensors it backpropagates
 * through stay resident.
 */
precomputed_features: boolean, };

export type StatsRequest = { path: string, 
/**
 * Only tensors whose name contains one of these (all if empty).
 */
only: Array<string>, 
/**
 * Compute K/V singular-value spectra per attention layer.
 */
kv_spectra: boolean, };

/**
 * One entry of the tensor index. No tensor data is held here.
 */
export type TensorInfo = { name: string, dtype: DType, 
/**
 * Logical shape in row-major order (outermost dimension first), matching
 * PyTorch/HF conventions. GGUF readers reverse ggml's `ne` order to match.
 */
shape: Array<number>, 
/**
 * Index into [`RawModel::files`].
 */
file: number, 
/**
 * Absolute byte offset of the tensor data within its file.
 */
offset: number, 
/**
 * Size of the tensor data in bytes.
 */
n_bytes: number, 
/**
 * False when `n_bytes` had to be inferred (e.g. an unknown GGUF type,
 * sized from the gap to the next tensor, which may include padding).
 */
bytes_exact: boolean, };

export type TensorKind = "attn_q" | "attn_k" | "attn_v" | "attn_qkv" | "attn_o" | "attn_gate" | "qk_norm" | "mla_q_a" | "mla_q_b" | "mla_kv_a" | "mla_kv_b" | "linear_attn" | "norm" | "ffn_gate" | "ffn_up" | "ffn_down" | "ffn_gate_up" | "router" | "expert" | "shared_expert" | "embedding" | "lm_head" | "other";

export type TensorRole = { component: Component, 
/**
 * Layer index for [`Component::Layer`] and layered MTP modules.
 */
layer: number | null, kind: TensorKind, };

export type TensorRow = { info: TensorInfo, role: TensorRole, };

export type TensorStats = { name: string, dtype: string, shape: Array<number>, kind: TensorKind, 
/**
 * Whether primal statistics were computed after undoing a folded rotation.
 */
unrotated: boolean, rms: number, mean: number, max_abs: number, 
/**
 * Excess kurtosis (0 for a Gaussian). Large values mean heavy outliers.
 */
kurtosis: number, 
/**
 * Largest input-channel RMS divided by the median input-channel RMS
 * (2-D weights only). Large values flag outlier channels.
 */
channel_outlier_ratio: number | null, zero_fraction: number, 
/**
 * Fraction of 128-wide row groups whose nonzero values share one magnitude.
 */
ternary_group_fraction: number | null, };

/**
 * How a stage uses the trunk. It decides the order stages run in: a stage
 * trained against the trunk is invalidated by any later change to it.
 */
export type TrunkUse = "restructure" | "adapt" | "frozen";

/**
 * An orthogonal rotation folded into the stored weights. Surgery has to keep
 * new and modified tensors in the same basis and keep the metadata in sync.
 */
export type WeightRotation = { scheme: string, version: number | null, block_size: number | null, sign_mode: SignMode, 
/**
 * Input widths that have their own sign vector.
 */
sign_widths: Array<number>, 
/**
 * Weight matrices stored in the rotated basis.
 */
rotated_tensors: number, 
/**
 * Tensors stored with the inverse rotation (e.g. input embeddings).
 */
inverse_tensors: number, 
/**
 * `ssm_out` inputs are kept in grouped (training) V-head order; the
 * runtime permutes activations from llama.cpp's tiled order first.
 */
gdn_v_grouped: boolean, 
/**
 * Metadata key prefix that declares the rotation.
 */
metadata_prefix: string, };

export type WeightStatsReport = { tensors: Array<TensorStats>, kv_spectra: Array<KvSpectrum>, skipped: Array<[string, string]>, notes: Array<string>, };
