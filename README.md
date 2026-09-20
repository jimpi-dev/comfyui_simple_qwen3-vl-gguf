<img width="2048" height="448" alt="15184-43452264163153+" src="https://github.com/user-attachments/assets/92b22216-aa55-4411-8718-8ec82e1b88b6" />

ComfyUI node for **Qwen3-VL** and other GGUF vision-language models.

The node talks to an external **llama.cpp `llama-server`** over HTTP. Python never loads GGUF files.

```text
ComfyUI
   │
   │ HTTP / OpenAI-compatible API
   ▼
llama-server   (-m Qwen3-VL-*.gguf --mmproj mmproj-*.gguf)
```

**No `llama-cpp-python` required.**

# Why this node?
1. GGUF models (faster than typical transformer loaders in ComfyUI).
2. Qwen3-VL, Qwen3.5, Gemma4 and other multimodal GGUFs that current llama.cpp already supports.
3. Easy to point at any new GGUF: start llama-server with `-m` / `--mmproj`, then set `server_url` on the node.
4. ComfyUI stays clean. The LLM and mmproj live in the llama-server process, so diffusion workflows are not fighting a Python-bound llama.cpp for VRAM.
5. No auto-downloaded models. Use GGUFs you already have (Hugging Face, LM Studio exports, etc.). Start llama-server with those files.

# Last update:

**4.0 — llama-server backend**

- Replaced `llama-cpp-python` with native `ggml-org/llama.cpp` `llama-server`.
- ComfyUI sends OpenAI-compatible `/v1/chat/completions` requests (images as base64 data URLs).
- Thinking uses llama-server `chat_template_kwargs.enable_thinking` / `reasoning_effort`.
- Model loading, GPU offload (`-ngl`), context (`-c`) and mmproj (`--mmproj`) are llama-server CLI flags, not Python bindings.

**Nightly (tests)**

- Add `streaming_mode`
- Add speculative decoding (now a llama-server CLI feature, not a Python SpecConfig)
- Add dynamic image, audio, video input, Add "user_prompt_template" input, Add "bypass" input
- New design for LLM Config
- **Added new configurator 🌐 LLM Config and 🌐 LLM Prompt Preset**

The advanced configurator still exposes sampling, prompts, media limits and (as documentation) llama-server CLI-related fields.

**Key Features:**

1. Built-in Preset Management - Direct access to JSON preset files from within ComfyUI.
2. Complete Parameter Access - Sampling, thinking, media and server URL in one place. Hardware/context widgets remain for documentation of the llama-server command line; they are not applied per HTTP request.
3. Windows File Browser - Browse buttons still help you note GGUF paths when writing a llama-server command.
4. Flexible Widget Layout - Adding new parameters does not corrupt old saves.

<img width="1122" height="590" alt="image" src="https://github.com/user-attachments/assets/c960ccd4-c400-448e-8def-45cbb301a327" />

> 💡 **TIP:** If you need to move preset lists to the top level of the subgraph, use widgets of the LLM Inference node — they work in the classic Comfy-UI way.

# Prerequisites

-------------
- ComfyUI
- [llama.cpp](https://github.com/ggml-org/llama.cpp) / `llama-server` (current official build)
- A Qwen3-VL (or other supported VL) GGUF **and** matching `mmproj` GGUF

**No llama-cpp-python required.**

Do not pin an old llama.cpp just because earlier versions of this node used Python bindings. Use a **current** `ggml-org/llama.cpp` release.

### Start llama-server (Linux / macOS)

```bash
llama-server \
  -m Qwen3-VL-8B-Instruct-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3-VL-8B-Instruct-F16.gguf \
  -ngl 99 \
  -c 8192 \
  --port 8080 \
  --jinja
```

### Start llama-server (Windows)

```bat
llama-server.exe ^
  -m Qwen3-VL-8B-Instruct-Q4_K_M.gguf ^
  --mmproj mmproj-Qwen3-VL-8B-Instruct-F16.gguf ^
  -ngl 99 ^
  -c 8192 ^
  --port 8080 ^
  --jinja
```

Useful flags (current llama.cpp):

| Flag | Meaning |
|------|---------|
| `-m` / `--model` | Text GGUF |
| `--mmproj` | Vision projector GGUF |
| `-ngl` / `--n-gpu-layers` | GPU offload (`99` or `all` = as many as fit) |
| `-c` / `--ctx-size` | Context size |
| `--port` | HTTP port (default `8080`) |
| `--jinja` | Use the GGUF chat template (recommended for Qwen3-VL thinking) |
| `--reasoning on\|off\|auto` | Default thinking behaviour for the whole server |
| `--no-mmproj-offload` | Keep the projector on CPU |
| `--image-min-tokens` / `--image-max-tokens` | Dynamic-resolution vision token limits |

Or download a ready-made pair, for example from [Qwen3-VL GGUF](https://huggingface.co/Qwen) / [ggml-org multimodal GGUFs](https://huggingface.co/collections/ggml-org/multimodal-ggufs-68244e01ff1f39e5bebeeedc).

Confirm the server is up:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

The ComfyUI node then posts to `http://127.0.0.1:8080/v1/chat/completions`.

Official docs:
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Multimodal: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- Server API: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

# Installation of ComfyUI_Simple_Qwen3-VL-gguf:
1.Installation to custom_nodes
- Use **ComfyUI Manager** and find **ComfyUI_Simple_Qwen3-VL-gguf**
- OR copy this project to the folder `path_to_comfyui\ComfyUI\custom_nodes`
```
  cd path_to_comfyui\ComfyUI\custom_nodes
  git clone https://github.com/KLL535/ComfyUI_Simple_Qwen3-VL-gguf
```
2. Restart ComfyUI. We check in the console that custom nodes are loading without errors.
3. Restarting the frontend (F5)

# Implementation Features:

ComfyUI never loads GGUF files. The node is an HTTP client:

```text
┌───────────────────────────────┐
│            ComfyUI            │
│  Qwen3-VL ComfyUI Node        │
│  No llama.cpp Python API      │
└───────────────┬───────────────┘
                │ HTTP  /v1/chat/completions
                ▼
┌───────────────────────────────┐
│         llama-server          │
│     current llama.cpp         │
│     Qwen3-VL GGUF + mmproj    │
└───────────────────────────────┘
```

Images are converted `ComfyUI IMAGE → JPEG → base64 data URL → OpenAI multimodal message`. Sampling (temperature, top-p, top-k, max tokens, stop, seed) is sent per request. GPU layers, context size and mmproj are set when you start llama-server.

# Nodes

🌐 SQVLM (Core):
- **🌐 LLM Inference (SQVLM)** - A universal Vision-Language model node supporting various GGUF models (Qwen, LLaVA, Gemma, MiniCPM, etc.).
- **🌐 LLM Config (Advanced)** *(NEW)* - The ultimate configuration node. Provides access to all 70+ supported parameters organized into collapsible, logical groups. Features built-in preset management (Save/Rename/Delete) and Windows file browsing.
- **🌐 LLM Prompt Preset** *(NEW)* - Similar to the previous node, it allows you to configure a collection of system prompts.

🛠️ Utils:
- **Master Prompt Loader** - Loads system prompt presets from JSON configuration files. Supports override via an optional string input. Ensures consistency across complex workflows.
- **Simple Style Selector** - Loads user prompt style presets. Can randomly select a style or apply a named preset, appending it to the user prompt for dynamic generation variation.
- **Simple Camera Selector** - Similar to Style Selector, but for camera-related descriptions (lens, lighting, angle). Appends photographic context to the user prompt.
- **Simple Qwen Unload** - Legacy pass-through. llama-server owns the GGUF; stopping the server (or router `POST /models/unload`) frees VRAM. Kept so old graphs still run.
- **Simple Remove Think** - Cleans model output by removing `<think>...</think>` sections. Designed for reasoning models (DeepSeek-R1, Qwen-thinking) to return only the final, cleaned response.
- **Simple Trigger Node** - Enforces execution order in complex workflows. Prevents heavy nodes (like `Load Checkpoint`) from executing prematurely and occupying VRAM unnecessarily.
- **Simple Text To Batch** - Splits LLM output by a given separator into a text batch, allowing you to extract multiple scenes or items from a single request.
- **Simple Text Insert** - Inserts text into a specific location defined by a placeholder.
- **Simple Text Replace** - Applies one or multiple rules for auto-replacement or deletion of words/phrases in a single node.
- **Simple Join Strings** - Concatenates up to 10 strings using a specified separator.
- **Ideogram 4 JSON Preview** - Visualizes bounding boxes from Ideogram 4 JSON output directly on the image.
- **Ideogram 4 JSON Swap XY Coordinates** - Fixes coordinate swapping (Y/X) for models like Qwen-9B that stubbornly ignore system instructions, preventing rotated bounding boxes.
- **Fix Batch Images** - Allows you to correctly assemble images into a batch even if some/all images are None.

📸 Video Utils
- **📸 Load Video Fragment** *(NEW)* - Extracts and processes a specific time-coded fragment from a large video file.
- **📸 Simple Gif Maker** *(NEW)* - Creates and saves GIFs with high compression optimization.

⚠️ Deprecated (Legacy):
- **Qwen-VL Vision Language Model** - Legacy version of the main node. Retained *only* for backward compatibility with old workflows. No longer actively developed.
- **LLM Model Config** - Legacy configuration node (Model parameters only).
- **LLM Sampling Config** - Legacy configuration node (Sampling parameters only).

# LLM Inference (SQVLM)
A universal version. The model and its parameters mast be passed to the `config_override` input or described in a file `ComfyUI/user/SimpleQwenVL_configs/system_prompts_user.json`

<img width="546" height="609" alt="image" src="https://github.com/user-attachments/assets/4e06cb5f-4901-4dc3-900d-1324e21806e0" />

<details>

<summary>Modes</summary>

| Mode | Characteristics | Benefits |
|--------|--------|--------|
| keep_vram (default) | HTTP to llama-server. The GGUF stays loaded in the server process. | Natural architecture. Repeated requests reuse the already-loaded model and mmproj. |
| subprocess / direct_clean / save1-save3 | Kept for existing workflows. Inference is still HTTP; these modes no longer load or unload a local GGUF. | Old graphs keep working. VRAM of the LLM is controlled by llama-server, not by this widget. |

> Stop llama-server when you need the GPU back for a heavy diffusion checkpoint. The ComfyUI node cannot unload the server-side model in single-model mode.


</details>

<details>

<summary>Parameters</summary>

### Parameters:
- `image`, `image2`, `image3`... (dynamic added inputs): *IMAGE* - The images to be analyzed. Batch processing is supported. Encoded as JPEG base64 data URLs for llama-server.
- `audio`, `audio2`, `audio3`... (dynamic added inputs): *AUDIO* - The audio files to be analyzed (loaded via Load Audio). 💡 Note: The **loaded llama-server model** must support audio (e.g. Gemma 4). See the audio_sample_rate parameter.
- `video`, `video2`, `video3`... (dynamic added inputs): *** - VIDEO — The video files to be analyzed (loaded via Load Video) or an image batch. By default the video is sent as a reduced set of JPEG frames (`max_frames`). Set `"native_video": true` in config to send llama-server `input_video` instead (ffmpeg on the server). 💡 Increase llama-server `-c` / `--ctx-size` for many frames.
- `model preset`: *LIST* - Selects a model based on templates defined in `system_prompts_user.json`.
- `system preset`: *LIST* - Selects a system prompt from predefined templates.
- `user prompt`: *STRING*, default: "Describe this image" - The specific prompt for the task, which can include input data and variable placeholders.
- `seed`: *INT*, default: 42
- `server_url`: *STRING*, default: `http://127.0.0.1:8080` - llama-server base URL. Optional `model` / `api_key` / `request_timeout` go in config.
- `unload_all_models`: *BOOLEAN*, default: false - If True, unloads **ComfyUI diffusion models** before the HTTP call. Does not stop llama-server.
- `mode`: *LIST*, default: `keep_vram` - Legacy widget. All values use HTTP against llama-server.
- `config override`: *STRING*, default: None - Overrides specific fields in the model preset template, or defines an entirely new model configuration if model preset is set to None. Example: `server_url: http://127.0.0.1:8080`
- `system prompt override`: *STRING*, default: None - If text is provided here, it will be used as the system prompt, and the **system preset will be ignored**.
- `user_prompt_template`: *STRING*, default: None - Allows you to set a custom user prompt template using {user_prompt} and other placeholders. When provided, automatic placeholder replacement is enabled.
- `variables`: *STRING*, default: None - Allows you to define custom user placeholders enclosed in curly braces {} for use in the system and user prompts. When provided, automatic placeholder replacement is enabled.
- `bypass`: *BOOLEAN*, default: false - If set to True, the node skips processing and passes the `user_prompt` directly to the output `text` without modification.
  
### Output:
- `text`: *STRING* - generated text
- `conditioning`: *CONDITIONING* - For embedding mode only
- `system preset`: *STRING* - Current system prompt (if you want to keep it)
- `user preset`: *STRING* - Current user prompt (if you want to keep it)

</details>

# Use Cases: 3 Ways to Configure Your Model

### Method 1: The New Advanced Configurator (Recommended)

The **🌐 LLM Config (Advanced)** node provides a clean, organized interface for all 70+ parameters. 
- Parameters are grouped into collapsible sections (Model, Memory, Sampling, Hardware, etc.), so you only see what you need.
- Windows users can use the "Browse" buttons to select GGUF files from anywhere on the disk.
- Outputs a ready-to-use JSON configuration string to the main node.

### Method 2: Model Preset Dropdown (Best for Workflow Reusability)

Once you have tuned your settings (either via the Advanced Configurator or manually), you can save them as a named preset.
- Use the **Save**, **Rename**, and **Delete** buttons in the Advanced Configurator to manage your library.
- Presets are saved to `system_prompts_user.json` in user folder.
- In the main node (or configurator), simply select your saved preset from the `model_preset` dropdown list. This instantly loads all associated parameters, making it easy to switch between different models without rewiring your workflow.

### Method 3: Manual Text Config (Best for Power Users & Stacking)

<img width="1331" height="696" alt="Image" src="https://github.com/user-attachments/assets/320192ed-d0c2-46bb-bc44-7f24d8348f3a" />

You can bypass the UI widgets entirely and pass configuration directly as a text string.
- `Flexible Formatting:` You don't need perfect JSON. If the `json_repair` library is installed, it will automatically fix missing commas or quotes.
- `Stacking & Overwriting:` Configurations are stackable. Each additional `config_override` input overwrites the specified fields and leaves the rest unchanged. 
- `Use Case:` This is the *only* way to pass brand-new, experimental parameters to the backend script before they are officially added to the Advanced Configurator's UI widgets.

> 💡 Pro Tip: You can combine all three methods! Set a base configuration using a `Preset`, tweak a few settings using the `Advanced Configurator`, and inject a final, specific override (like a custom `stop` sequence) via the `config_override` text input. The system resolves them in that exact order of priority.

# Configurations

Possible model configurations that can be passed to the `config_override` input.

<details>

<summary>Configurator Parameters</summary>

📁 Model & Paths

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| model_preset | dropdown | None | Select from saved model presets. Presets are loaded from `system_prompts_user.json` |
| server_url | string | http://127.0.0.1:8080 | llama-server base URL. The node POSTs to `{server_url}/v1/chat/completions` |
| api_key | string | "" | Optional. Only needed if llama-server was started with `--api-key` |
| request_timeout | int | 300 | HTTP timeout in seconds |
| model | string | "" | Optional OpenAI model name. Empty = the model already loaded by llama-server (or basename of `model_path` in router mode) |
| model_path | string | "" | Optional GGUF path **hint** / router model id. This node does **not** load the file. Start llama-server with `-m` yourself |
| mmproj_path | string | "" | Optional mmproj path hint. Load it on llama-server with `--mmproj` |

> Hardware (`n_gpu_layers`, `n_ctx`, `n_batch`, `cpu_moe`, …) and speculative-decoding widgets are **llama-server CLI documentation**. They are not applied per HTTP request. Set them when starting `llama-server`.

🗄️ Memory & Context

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| n_ctx | int | 8192 | Context size (max tokens model can process). Rule: `image_tokens + input_tokens + max_tokens ≤ n_ctx`. Increasing this increases VRAM consumption. Too small = truncated responses |
| n_batch | int | 2048 | Batch size for prompt processing. Lower = less VRAM, higher = faster prompt evaluation. Setting `n_batch = n_ctx` can speed up processing |
| n_ubatch | int | 512 | 	Micro-batch size for advanced memory management. Controls physical batch size during inference |
| n_keep | int | 256 | 	Number of tokens to keep in KV-cache from initial prompt. Useful for few-shot/long-context scenarios |
| offload_kqv | bool | True | Offload KV Cache to GPU. Turn OFF to save VRAM (will be slower). Prevents VRAM overflow |
| type_k | dropdown/int | 1=F16 | KV-cache quantization type for Keys. Controls compression/quantization level. 💡 Some variants may not work with all model | 
| type_v | dropdown/int | 1=F16 | KV-cache quantization type for Values. Same as type_k but for Value tensors | 
| use_mmap | bool | False | Enable memory mapping for model loading. 💡 On Windows, it's often better to turn OFF for stability |
| use_mlock | bool | False | Enable mlock. Lock model in RAM to prevent OS swapping. Uses more RAM but prevents page faults |
| pool_size | int | 4194304 | Memory pool size for llama.cpp. Increase if you get ggml_new_object: not enough space |
| logits_all | bool | False | Evaluate logits for ALL tokens (not just last one). Required for perplexity evaluation, but significantly increases VRAM and time |
| swa_full | bool | False | Enable full Sliding Window Attention context. Required for some models (Mistral/Gemma) to prevent truncation |

🎲 Sampling & Generation
| Field | Type | Default | Description |
|--------|--------|--------|--------|
| max_tokens | int | 2048 | Maximum tokens to generate. Thinking models usually need more (4096+). Smaller = faster but may truncate response |
| temperature | float | 0.7 | Sampling temperature. Lower (0.1) = deterministic/focused, Higher (1.5+) = creative/random. 0.7 is balanced |
| top_p | float | 0.92 | Nucleus sampling cutoff. Model considers tokens whose cumulative probability reaches top_p. Lower = more focused |
| min_p | float | 0.05 | Minimum probability threshold. Tokens with prob < min_p × top_token_prob are filtered out. Great for reducing garbage |
| top_k | int | 0 | Limit to top-K most likely tokens. 0 = disabled. Good for strict output control |
| repeat_penalty | float | 1.1 | Penalty for repeating tokens. Values >1 discourage repetition loops. 1.1 is mild, 1.5+ is aggressive |
| presence_penalty | float | 0.0 | Penalty based on token presence. Positive values encourage new topics, negative favor repetition |
| frequency_penalty | float | 0.0 | Penalty based on token frequency. Positive values reduce repetition of common words |
| enable_thinking | bool | False | Enable thinking. Sent as `chat_template_kwargs.enable_thinking` and (when off) `reasoning_effort: "none"`. The GGUF Jinja template decides the exact tags |
| remove_thinking | bool | False | Cleans model output by removing `<think>...</think>` or reasoning_content |
| force_reasoning | bool | False | Same as enable_thinking=true for Qwen3-VL. Forces reasoning even on simple queries |
| words_to_ban | string | "" | Comma-separated list of banned words. Applies logit_bias of -100 to their tokens. Example: woman,Woman,man,Man |

⚙️ Hardware & Acceleration

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| n_gpu_layers | int | -1 | Layers to offload to GPU. -1 = all, 0 = CPU only. Reduce if OOM (try 40→35→30) |
| n_cpu_moe | int | 0 | For MoE models: experts to keep on CPU. Saves VRAM. Slower than full GPU, but faster/stable than OS swap |
| cpu_moe | bool | False | For MoE models: unload ALL experts into RAM. Minimal VRAM usage, slower inference |
| n_threads | int | 8 | CPU threads for inference. Match physical cores (not hyperthreads) for best performance |
| flash_attn_type | dropdown/int | -1=AUTO | Flash Attention backend. Requires compatible llama.cpp build. AUTO selects best available |
| split_mode | dropdown/int | 0-NONE | GPU splitting: 0=NONE (single GPU), 1=LAYER (distribute layers), 2=ROW (tensor parallelism) |
| main_gpu | int | 0 | Primary GPU index when split_mode=NONE. Works with CUDA_VISIBLE_DEVICES filtering |
| cuda_device | string | "" | Sets CUDA_VISIBLE_DEVICES before init. Single index (0) or comma-separated (0,1). Empty = not set |
| tensor_split | list of strings | "" | Fractions for GPU split (e.g., [0.7, 0.3] for 70%/30%). Only for split_mode=LAYER. Empty = auto-balance |

💬 Chat, Prompts & Variables

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| chat_handler | dropdown/string | "none" | Legacy llama-cpp-python handler name. **Ignored.** llama-server uses the Jinja chat template stored in the GGUF |
| chat_format | dropdown/string | "none" | Chat format for text-only models: llama-2, llama-3, chatml, alpaca, etc. Not needed if chat_handler is set |
| chat_format_from_gguf | bool | False | Force loading chat template from GGUF metadata. 💡 Does NOT work with images/audio/video |
| system_prompt_default | string | "" | Default system prompt for the model. Used when no preset or override is provided | 
| system_preset_to_user_prompt | bool | False | Move system preset from system prompt role to user prompt role. Useful for models that follow user prompts better | 
| user_prompt_after_content | bool | True | Insert user_prompt AFTER image/audio/video content. False = insert before |
| enable_variables | bool | False | Enable substitution of {placeholders} in system and user prompts. Auto-vars: {image_num}, {width}, {height}, etc. |
| add_vision_id | dropdown/int | "auto" | Add vision ID token. auto = script decides (True if images ≠ 1 or video > 0). Required for Qwen3/Qwen3.5 |
| add_image_id | string | "" | Template to label images: `\n[Image {num}]:`. {num} = image index. Helps model distinguish multiple images | 
| add_frame_id | string | "" | Template to label video frames: `\n[Frame {num}]:`. Useful for video understanding tasks | 
| add_audio_id | string | "" | Template to label audio files: `\n[Audio {num}]:`. For multi-audio scenarios | 

💬 Prompt Template

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| raw_mode | bool | False | Enable custom raw prompt template mode (bypasses chat handlers). Required for custom templates |
| prompt_template | string | "" | Custom prompt template. Must include {system}, {images}, {user}. |
| stop | list of strings | "" | Stop sequences. JSON list ["</s>", "[INST]"] or comma-separated. Empty = handler default |

🖼️ Multimodal & Media

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| force_mmproj | bool | True | Load mmproj even without media inputs. Preserves template for enable_thinking. Uses VRAM unnecessarily if no media |
| image_min_tokens | int | 0 | Minimum tokens for image embeddings. 0 = not set. Controls memory allocation |
| image_max_tokens | int | 0 | Maximum tokens for image embeddings. 0 = not set. Prevents oversized image encodings |
| max_images | int | 10 | Limit on total incoming images across image/image2/image3 inputs (batch mode can send many) | 
| max_frames | int | 24 | Limit on video frames. More frames = larger context needed. Scaling may lose motion details | 
| max_audios | int | 3 | Limit on incoming audio clips. Batch mode can send multiple audio per input | 
| audio_sample_rate | int | 0 | Target sampling frequency for audio resampling. 0 = not set (keep original) | 
| image_quality | int | 95 | JPEG quality (1-100) when encoding images to data URIs. Higher = better quality, larger size |
| frame_quality | int | 75 | JPEG quality (1-100) when encoding video frames. Lower than images to save space |

⚡ Speculative Decoding

Speculative decoding accelerates text generation by using a draft model (or statistical n-gram) to predict multiple tokens ahead, which are then verified by the target model in a single batch pass. This can significantly speed up inference when the draft predictions are accurate.

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| speculative_enabled | bool | False | Master switch to enable speculative decoding. |
| speculative_type | int | 3=MTP | Speculative algorithm type. `3=MTP` (Multi-token Prediction, built-in for Qwen3.5/3.8, recommended), `4=DFLASH` (Block-diffusion draft, requires external model), `5=DSPARK` (Markov/confidence heads, requires external model), `7=NGRAM_MAP_K` (Statistical n-gram, no draft model needed, good for code/JSON), `8=NGRAM_MAP_K4V` (N-gram with 4 cached continuations per key). Other types (1,2,6,9,10) are experimental or legacy. |
| draft_n_max | int | 2 | Maximum number of draft tokens to generate per step. `Recommended: 2 for MTP`, `7 for DFlash/DSpark`. Higher values increase potential speedup but reduce acceptance rate. Must be ≤ `n_batch - 1`. |
| draft_p_min | float | 0.0 | Minimum probability threshold to accept a draft token. `0.0` = accept all. For `DFlash/DFlash2`, filters transition probability. For DSpark, filters acceptance confidence. |
| draft_model_path | string | "" | Path to external draft GGUF model. Required for DFlash (4) and DSpark (5). Leave empty for built-in MTP (3) or N-gram (7/8). The draft model vocabulary and embedding dimensions must match the target model. |
| draft_n_gpu_layers | int | -1 | **[External model only]** Number of layers to offload for the external draft model. `-1` = all layers on GPU, `0` = CPU only. Only used when `draft_model_path` is specified. |
| draft_backend_sampling | bool | True | **[External model only]** Use backend vocabulary sampler for draft tokens. Recommended `True` for `DFlash v1` and `DSpark` with large vocabularies. `DFlash2` reads its compact selector output directly and ignores this setting. |
| ngram_size_n | int | 8 | **[N-gram only]** Size of the n-gram window (N).** Defines how many previous tokens to match when searching for continuations in the generated text history. |
| ngram_size_m | int | 16 | **[N-gram only]** Maximum length of the draft continuation (M).** How many tokens ahead to propose when a matching n-gram pattern is found. Longer drafts can be faster for highly repetitive output (code, JSON, templates). |
| ngram_min_hits | int | 1 | **[N-gram only]** Minimum number of matching occurrences required to propose a draft.** Higher values increase confidence but reduce the number of proposals. |
| ngram_max_entries_per_key | int | 4 | **[NGRAM_MAP_K4V only]** Maximum cached continuations per n-gram key.** Only used when `speculative_type=8`. Allows caching multiple possible continuations for each n-gram pattern. |
| ctx_checkpoints | int | 0 | Max number of context checkpoints per slot for rollback support. Set to 16 if using N-gram speculative decoding (required for rollbacks when draft is rejected). For standard 1-question-1-answer generation or MTP/DFlash methods, keep at `0` to save VRAM. |
| checkpoint_on_device | bool | False | Store context checkpoints in VRAM (`True`) instead of RAM (`False`). Saves VRAM if `False`, but makes rollbacks slower due to PCIe transfer. Only matters if `ctx_checkpoints > 0` (i.e., only for N-gram). |

🔢 Embeddings

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| extract_embedding | bool | False | Switch to embedding extraction mode. Calls llama-server `POST /embedding` (fallback `/v1/embeddings`). Text output replaced by CONDITIONING tensor |
| pooling_type | dropdown/int | 0-NONE | Pooling strategy: -1=UNSPECIFIED (auto), 0=NONE (per-token), 1=MEAN (average), 2=CLS (first token), 3=LAST (last token), 4=RANK (reranking) |
| tokenizer_path | string | "" | Ignored. Embeddings use the tokenizer inside llama-server |
| embedding_scale | float | 1.0 | Scalar multiplier for output embedding vector. 1.0 = no scaling. Match magnitude for downstream models |
| convert_emb_to_cond | bool | False | Wrap embedding into ComfyUI CONDITIONING (hidden_states + attention_mask). Required for SD/Flux conditioning |

🛠️ Debug, System & Advanced

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| verbose | bool | False | Enable verbose logging from llama.cpp. Prints detailed inference info to console |
| debug | bool | True | Enable timing output for each stage in console. Shows metrics [DEBUG] inference 80.11 tok/sec 1812 tokens: 22.619s |
| debug_output | bool | False | Print final LLM text output to console | 
| raw_output | bool | False | Disable output.strip(). Keeps leading/trailing whitespaces in response | 
| streaming_mode | bool | False | Enables token streaming to allow interrupting generation via the ComfyUI 'Interrupt' button. Adds a negligible overhead (~1%), but guarantees you can manually stop long responses. Recommended if you often need to cancel generations. Ignore in subprocess mode | 
| clearing_cache | bool | True | Clear cache to prevent execution freezing during heavy memory activity | 
| force_gc_start | bool | False | Force garbage collection after memory clearing (when unload_all_models active). Increases time but cleans memory | 
| force_gc_unload | bool | False | Force garbage collection after deleting LLM model. Prevents memory leake | 
| script | string | "qwen3vl_run.py" | Name of Python script to execute. Usually don't need to change |
| extra | string | "" | JSON dict of extra keys passed to backend script. For advanced custom parameters |

**Notes & Nuances**

1. Browse Button Limitation
The Browse Model and Browse MMProj buttons currently work only on Windows (using native file dialog via ctypes). Linux/macOS users must manually type paths. If there's demand, I can implement GTK/Qt dialogs for other platforms.

2. Override Input Behavior
The config_override input strictly overwrites fields passed through it. This means:
- Values shown in widgets may differ from actual output if override is used
- Override has highest priority (applied last)
- Use override for dynamic/runtime changes, widgets for static defaults

3. Parameter Naming Consistency
All parameters use canonical names (n_ctx, n_gpu_layers, max_tokens). Old names (ctx, gpu_layers, output_max_tokens) are automatically converted via old_names_patch() for backward compatibility.

4. Widget Reordering
Parameters are rendered in a fixed order matching the Python node's INPUT_TYPES(). However, the underlying architecture supports reordering via **kwargs, so future versions may allow custom layouts without breaking saved workflows.

5. Preset Storage
Presets are saved to ComfyUI/user/SimpleQwenVL_configs/system_prompts_user.json. The file is created automatically on first use. 

6. Multi-GPU Caveats
cuda_device parameter may not work correctly in direct_clean and keep_vram modes, as ComfyUI itself may have already initialized CUDA with different settings.

7. Vision ID Logic
add_vision_id with auto mode calculates: True if (num_images != 1 or num_videos > 0) else False. This matches Qwen3/Qwen3.5 requirements for multi-image scenarios.

8. Memory Pool Sizing
pool_size default (4194304 = 4MB) works for most models. If you encounter ggml_new_object: not enough space, increase to 8MB (8388608) or 16MB (16777216).

</details>

## Configuration Files & Presets

<details>
  
<summary>Rules & File Hierarchy</summary>

The system uses a stackable configuration approach. Files are loaded in the following order of priority:

1. `ComfyUI/user/SimpleQwenVL_configs/system_prompts_user.json` *(Recommended)*  
   This is the primary user settings file. It is created automatically on first use. The new `Advanced Configurator` reads from and writes to this file directly via its Save/Rename/Delete buttons. *Edit this file or manage it via the UI.*

2. `system_prompts_user.json` *(Legacy Node Folder)*  
   Located in the node's root directory. Supported for backward compatibility with older setups. If both this file and the `user/` directory file exist, the `user/` directory file takes precedence. Manual editing is discouraged in favor of the UI manager.

3. `system_prompts.json` *(Base Project Settings)*  
   Located in the node's root directory. Contains default, project-level presets maintained by the developer. **Do not edit this file**, as your changes will be overwritten during node updates.

</details>

## User variables input (plaseholders):

<details>
  
<summary>User variables input</summary>

Any placeholders { } can now be specified in the system and user prompts. Their values ​​can be determined through the variables input. Moreover, some of them, if not specified by the user, will be automatically inserted:
{width}, {height}, {image_num}, {ref_num}, {audio_num}, {frame_num}, {user_prompt}

Where:
- `{image_num}` - The total number of images fed to inputs image, image2, image3 at most `max_images` (default 10, see `max_images` config).
- `{frame_num}` - The total number of frames fed to input video at most `max_frames` (default 25, see `max_frames` config).
- `{audio_num}` - The total number of audios fed to input audio at most `max_audios` (default 3, see `max_audios` config).
- `{ref_num}` = {image_num}-1. This is needed for instructions where there is one base image (image input), and the rest are reference images.
- `{user_prompt}` - Text from the user_prompt input
- `{width}` - Length of the first image
- `{height}` - Height of the first image

> 💡 **WARNING:** By default, placeholder replase is disabled for backward compatibility. It can be enabled by:
> - passing user variables to the `variables` input (just like the config input)
> - by using `_user_prompt_template`
> - by forcing it by entering `"enable_variables": true,` in config.

<img width="1199" height="660" alt="Image" src="https://github.com/user-attachments/assets/a5923aa8-3733-4464-9383-60a571dfdf10" />

</details>

# Utils

Description of additional utilities

<details>

<summary>Utils</summary>

## Master Prompt Loader

Allows select a system prompt from templates. In the simplified version of LLM this switch is built in.
<img width="602" height="245" alt="image" src="https://github.com/user-attachments/assets/fbe21fb5-3e9b-4ddc-872f-c722de8190fc" />

<details>

<summary>Parameters</summary>

### Parameters:
- `system prompt opt`: *STRING* - input user text (postfix)
- `system preset`: *LIST* - allows you to select a system prompt from templates

### Output:
- `system prompt`: *STRING* - output = system prompt + input user text, connect to LLM system_prompt input

</details>

## Simple Style Selector/Simple Camera Selector
Allows select a user prompt from templates:
- Styles - replacing an image style, work well.
- Camera settings - instruction to describe the camera, can sometimes give interesting results.

<img width="932" height="240" alt="image" src="https://github.com/user-attachments/assets/53278c09-71f7-4775-a6d1-75c7f909fef1" />

<details>

<summary>Parameters</summary>

### Parameters:
- `user prompt`: *STRING* - input user text (prefix)
- `style/camera preset`: *LIST* - allows you to select a style/camera templates

### Output:
- `user prompt`: *STRING* - output = input user text + style/camera prompt, connect to LLM user_prompt input
- `style/camera name`: *STRING* - preset name (if you want to keep it)

</details>

</details>

# Models (for example):

<img width="2048" height="448" alt="03522-929995336568847" src="https://github.com/user-attachments/assets/0dc6c148-c049-4fc4-9363-eedb04db2785" />

<details>

<summary>Qwen3.8-27B</summary>

- https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF

For example (for 16 Gb VRAM):
`Qwen3.8-27B-ABLITERATED-Q3_K_S.gguf` + `mmproj-BF16.gguf`

> 💡 **WARNING:** Parameters not specified in this list have **default** values. See `Model Configs` sections.

```json
{
    "model_path": "I:\\LLM\\qwen\\Qwen3.8-27B\\Qwen3.8-27B-ABLITERATED-Q3_K_S.gguf",
    "mmproj_path": "I:\\LLM\\qwen\\Qwen3.8-27B\\mmproj-BF16.gguf",
    "n_batch": 4096,
    "top_p": 0.8,
    "top_k": 20,
    "repeat_penalty": 1.05,
    "chat_handler": "qwen35",
    "image_min_tokens": 1024,
    "image_max_tokens": 2048
}
```

</details>

<details>

<summary>Ernie Image Prompt Enhancer</summary>

Highly specialized LLM for Ernie Image. 

- https://huggingface.co/Green-Sky/Ernie-Image-Prompt-Enhancer-Ministral-3B-GGUF

- https://huggingface.co/unsloth/Ministral-3-3B-Reasoning-2512-GGUF

For example: `Ernie-Image-Prompt-Enhancer-Ministral-3.8B-Q4_K_M.gguf` + `mmproj-BF16.gguf`

> 💡 **TIP:** A special `system prompt` is required; in the templates it is called `Ernie Prompt Enhancer`.

> 💡 **TIP:** The `user prompt` should look like this: `{"prompt": "{prompt}", "width": {width}, "height": {height}}`

> 💡 **TIP:** `mmproj` should be left empty "" if image input are not needed.

> 💡 **TIP:** The result will only be in Chinese.

```json
{
    "model_path": "H:\\LLM3\\ernie\\Ernie-Image-Prompt-Enhancer-Ministral-3.8B-Q4_K_M.gguf",
    "mmproj_path": "H:\\LLM3\\ernie\\mmproj-BF16.gguf",
    "n_ctx": 4096,
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": 64,
    "repeat_penalty": 1.05,
    "chat_handler": "llava15",
    "raw_mode": true,
    "prompt_template": "[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user}{images}[/INST]",
    "stop": "[\"</s>\",\"[INST]\",\"[/INST]\"]",
    "image_min_tokens": 1024,
    "image_max_tokens": 1024
}
```

</details>

<details>

<summary>gemma-4-12B</summary>

- https://huggingface.co/lmstudio-community/gemma-4-12B-it-QAT-GGUF

```json
{
    "model_path": "I:\\LLM\\gemma\\gemma-4-12B-it-QAT-GGUF\\gemma-4-12B-it-qat-uncensored-heretic-UDmerge-Q4_K_XXL.gguf",
    "mmproj_path": "I:\\LLM\\gemma\\gemma-4-12B-it-QAT-GGUF\\mmproj-gemma-4-12B-it-QAT-BF16.gguf",
    "n_ctx": 12288,
    "max_tokens": 10240,
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 20,
    "repeat_penalty": 1.05,
    "chat_handler": "gemma4",
    "enable_thinking": true
}
```

For Audio/Video:
```json
{
    "model_path": "I:\\LLM\\gemma\\gemma-4-12B-it-QAT-GGUF\\gemma-4-12B-it-qat-uncensored-heretic-UDmerge-Q4_K_XXL.gguf",
    "mmproj_path": "I:\\LLM\\gemma\\gemma-4-12B-it-QAT-GGUF\\mmproj-gemma-4-12B-it-QAT-BF16.gguf",
    "n_ctx": 80384,
    "n_batch": 512,
    "max_tokens": 10240,
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 20,
    "repeat_penalty": 1.05,
    "chat_handler": "gemma4",
    "enable_thinking": true,
    "max_frames": 120,
    "audio_sample_rate": 16000
}
```

</details>

<details>

<summary>HY-MT2 (translate)</summary>

- https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF

For example: `Hy-MT2-1.8B-Q4_K_M.gguf`

> 💡 **TIP:** Here I made a prompt template in which the target_language is set through the `system_prompt_override` input. Just supply the text with the target language there, for example `Russian`. And the text that needs to be translated should be submitted to the `user_prompt` input.

> 💡 **WARNING:** The model is highly specialized and understands only strictly defined tasks.

```json
{
    "model_path": "H:\\LLM3\\Hy-MT2-1.8B-Q4_K_M.gguf",
    "n_ctx": 4096,
    "n_batch": 4096,
    "use_mmap": true,
    "top_p": 0.6,
    "top_k": 20,
    "repeat_penalty": 1.05,
    "system_prompt_default": "Russian",
    "raw_mode": true,
    "prompt_template": "<｜hy_begin▁of▁sentence｜>Translate the following segment into {system}, without additional explanation.<｜hy_place▁holder▁no▁3｜><｜hy_User｜>{user}<｜hy_Assistant｜>",
    "stop": "[\"<｜hy_place▁holder▁no▁2｜>\"]"
}
```

- https://huggingface.co/tencent/Hy-MT2-7B-GGUF

For example: `Hy-MT2-7B-Q4_K_M.gguf`

```
{
    "model_path": "H:\\LLM3\\Hy-MT2-7B-Q4_K_M.gguf",
    "n_ctx": 4096,
    "n_batch": 4096,
    "use_mmap": true,
    "top_p": 0.6,
    "top_k": 20,
    "repeat_penalty": 1.05,
    "system_prompt_default": "Russian",
    "raw_mode": true,
    "prompt_template": "<|startoftext|>Translate the following segment into {system}, without additional explanation.<|extra_4|>{user}<|extra_0|>",
    "stop": "[\"<|eos|>\"]"
}
```

- https://huggingface.co/mradermacher/Hy-MT2-30B-A3B-GGUF

For example: `Hy-MT2-30B-A3B.Q4_K_M.gguf`

> 💡 **TIP:** "n_cpu_moe": 12 to 16G VRAM

```
{
    "model_path": "H:\\LLM3\\Hy-MT2-30B-A3B-Q4_K_M.gguf",
    "n_ctx": 4096,
    "n_batch": 4096,
    "temperature": 0.3,
    "top_p": 0.6,
    "top_k": 20,
    "repeat_penalty": 1.05,
    "n_cpu_moe": 12,
    "system_prompt_default": "Russian",
    "raw_mode": true,
    "prompt_template": "<|start_header_id|>user<|end_header_id|>\\n\\nTranslate the following segment into {system}, without additional explanation.\\n\\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n",
    "stop": "[\"<|eot_id|>\",\"<|start_header_id|>\"]"
}
```

</details>

<details>

<summary>Qwen3.6-35B-A3B</summary>

- https://lmstudio.ai/models/qwen/qwen3.6-35b-a3b

For example:
`Qwen3.6-35B-A3B-Q4_K_M.gguf` + `mmproj-Qwen3.6-35B-A3B-BF16.gguf`

> 💡 **Tip:** Q4_K_M is already quite an old quantization. Search for models on huggingface and choose models with better quantization, such as UD_IQ from unsloth. They will be smarter and lighter.

- https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF

For example:
`Qwen3.6-35B-A3B-APEX-I-Quality.gguf` + `mmproj.gguf`

- https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF

For example:
`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` + `mmproj-BF16.gguf`

> 💡 **Tip:** If there is a BF16 version for mmproj, choose it, it is better than F16.


Examples:

This model not fit in 16 Gb VRAM.
Settings for `n_cpu_moe` offloading:

> 💡 **Tip:** `use_mmap = false` - Provides better speed, but the model may take longer to load, it needs to be tested.

> 💡 **Tip:** `split_mode = 0` - Provides better speed on a single GPU, eliminating performance drops after launch.

```json
{
    "model_path": "H:\\LLM\\lmstudio-community\\Qwen3.6-35B-A3B-GGUF\\Qwen3.6-35B-A3B-Q4_K_M.gguf",
    "mmproj_path": "H:\\LLM\\lmstudio-community\\Qwen3.6-35B-A3B-GGUF\\mmproj-Qwen3.6-35B-A3B-BF16.gguf",
    "use_mmap": true,
    "max_tokens": 4096,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "n_cpu_moe": 20,
    "chat_handler": "qwen35",
    "enable_thinking": true,
    "image_min_tokens": 1024,
    "image_max_tokens": 2048
}
```

```json
{
    "model_path": "H:\\LLM2\\qwen\\Qwen3.6-35B-A3B-UD\\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
    "mmproj_path": "H:\\LLM2\\qwen\\Qwen3.6-35B-A3B-UD\\mmproj-BF16.gguf",
    "use_mmap": true,
    "max_tokens": 4096,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "n_cpu_moe": 16,
    "chat_handler": "qwen35",
    "enable_thinking": true,
    "image_min_tokens": 1024,
    "image_max_tokens": 2048
}
```

```json
{
    "model_path": "H:\\LLM2\\qwen\\Qwen3.6-35B-A3B-APEX\\Qwen3.6-35B-A3B-APEX-I-Quality.gguf",
    "mmproj_path": "H:\\LLM2\\qwen\\Qwen3.6-35B-A3B-APEX\\mmproj.gguf",
    "use_mmap": true,
    "max_tokens": 4096,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "n_cpu_moe": 20,
    "chat_handler": "qwen35",
    "enable_thinking": true,
    "image_min_tokens": 1024,
    "image_max_tokens": 2048
}
```

</details>

<details>

<summary>Qwen3.6-27B</summary>

- https://huggingface.co/unsloth/Qwen3.6-27B-GGUF

For example:
`Qwen3.6-27B-UD-IQ3_XXS.gguf` + `mmproj-BF16.gguf`

Fit in 16 Gb VRAM:

```json
{
    "model_path": "H:\\LLM\\lmstudio-community\\Qwen3.6-27B-GGUF\\Qwen3.6-27B-UD-IQ3_XXS.gguf",
    "mmproj_path": "H:\\LLM\\lmstudio-community\\Qwen3.6-27B-GGUF\\mmproj-BF16.gguf",
    "top_p": 0.95,
    "top_k": 40,
    "chat_handler": "qwen35",
    "enable_thinking": true,
    "image_min_tokens": 1024,
    "image_max_tokens": 2048
}
```

</details>

<details>

<summary>Nemotron-3-Nano-Omni-30B</summary>

- https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF

For example:
`NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-IQ4_NL.gguf` + `mmproj-BF16.gguf`

Not fit in 16 Gb VRAM -> Use `n_cpu_moe = 24`:

```json
{
    "model_path": "H:\\LLM2\\nemotron\\NVIDIA-Nemotron-3-Nano-Omni-30B\\NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-IQ4_NL.gguf",
    "mmproj_path": "H:\\LLM2\\nemotron\\NVIDIA-Nemotron-3-Nano-Omni-30B\\mmproj-BF16.gguf",
    "n_batch": 8192,
    "max_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 40,
    "n_cpu_moe": 24,
    "chat_handler": "qwen35",
    "enable_thinking": true
}
```

> 💡 **WARNING:** Chat handler `qwen35` is not compatible with this model. The model will work, but the quality may deteriorate.

There is an alternative solution: override the chat template.
If you only need to process text and/or images, you can use this template overrides:
Thinking version (add these lines):

```json
    "chat_handler": "llava15",
    "raw_mode": true,
    "prompt_template": "<|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{images}{user}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n",
    "stop": "<|endoftext|>"
```

Non-thinking version (add these lines):

```json
    "chat_handler": "llava15",
    "raw_mode": true,
    "prompt_template": "<|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{images}{user}<|im_end|>\\n<|im_start|>assistant\\n",
    "stop": "<|endoftext|>"
```

</details>

<details>

<summary>Gemma4-26B-A4B</summary>

- https://huggingface.co/noctrex/gemma-4-26B-A4B-it-uncensored-heretic-MXFP4_MOE-GGUF

For example:
`Huihui-gemma-4-26B-A4B-it-abliterated-MXFP4_MOE.gguf` + `mmproj-BF16.gguf`

Not fit in 16 Gb VRAM -> set `n_cpu_moe`.

```json
{
    "model_path": "H:\\LLM2\\gemma\\Huihui-gemma-4-26B-A4B-it-abliterated-MXFP4_MOE\\Huihui-gemma-4-26B-A4B-it-abliterated-MXFP4_MOE.gguf",
    "mmproj_path": "H:\\LLM2\\gemma\\Huihui-gemma-4-26B-A4B-it-abliterated-MXFP4_MOE\\mmproj-BF16.gguf",
    "n_ctx": 4096,
    "n_batch": 512,
    "max_tokens": 4096,
    "top_p": 0.95,
    "top_k": 40,
    "n_cpu_moe": 10,
    "chat_handler": "gemma4"
}
```

</details>

<details>

<summary>Gemma4-E4B</summary>

- https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF
- https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF
- https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive

For example:
`gemma-4-E4B-it-IQ4_XS.gguf` + `mmproj-BF16.gguf`

Option appeared `enable_thinking": false`, but he doesn't turn off thinking :).

```json
{
    "model_path": "H:\\LLM2\\gemma4\\gemma-4-E4B-it-IQ4_XS.gguf",
    "mmproj_path": "H:\\LLM2\\gemma4\\mmproj-BF16.gguf",
    "n_ubatch": 2048,
    "temperature": 1.0,
    "top_p": 0.95,
    "min_p": 0.01,
    "top_k": 64,
    "repeat_penalty": 1.0,
    "chat_handler": "gemma4"
}
```

You can write custom `prompt template` and then thinking will turn off.

```json
{
    "model_path": "H:\\LLM2\\gemma4\\gemma-4-E4B-it-IQ4_XS.gguf",
    "mmproj_path": "H:\\LLM2\\gemma4\\mmproj-BF16.gguf",
    "n_ubatch": 2048,
    "temperature": 1.0,
    "top_p": 0.95,
    "min_p": 0.01,
    "top_k": 64,
    "repeat_penalty": 1.0,
    "chat_handler": "gemma4",
    "raw_mode": true,
    "prompt_template": "<|turn>system\n{system}<turn|>\n<|turn>user\n{images}\n{user}<turn|>\n<|turn>model\n",
    "stop": "[\"<turn|>\",\"<eos>\",\"<|end_of_turn|>\"]"
}
```

</details>

<details>

<summary>Sulphur prompt enhancer</summary>

An interesting uncensored fine-tuned model for LTX 2.3.

- https://huggingface.co/SulphurAI/Sulphur-2-base/tree/main/prompt_enhancer

> 💡 **Warning:** A highly specialized model for enhance prompts for LTX 2.3.

> 💡 **Warning:** The module is poorly described, so the following settings are set by eye. More optimal settings may exist.

> 💡 **Tip:** `system_preset_to_user_prompt: true` means that the system prompt will be passed to the user prompt (before user prompt). 

> 💡 **Tip:** `user_prompt_after_content: false` means that the image will be transmitted at the end.

system_prompt: `none` or `LTX I2V` or `LTX T2V` or `enhance this for video generation`

```json
{
    "model_path": "H:\\LLM2\\sulphur\\sulphur_prompt_enhancer_model-q8_0.gguf",
    "mmproj_path": "H:\\LLM2\\sulphur\\mmproj-BF16.gguf",
    "n_batch": 4096,
    "use_mmap": true,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "chat_handler": "qwen35",
    "system_preset_to_user_prompt": true,
    "user_prompt_after_content": false,
    "force_mmproj": false,
    "image_min_tokens": 1024,
    "image_max_tokens": 2048
}
```

</details>

<details>

<summary>Cydonia-24B</summary>

An interesting fine-tuned model based on mistral.

- https://huggingface.co/mradermacher/Cydonia-24B-v4.3-absolute-heresy-GGUF

There is no visual encoder (mmproj) here, but you can take it from the base model (Mistral-Small), for example from here:

- https://huggingface.co/ggml-org/Mistral-Small-3.1-24B-Instruct-2503-GGUF/tree/main

> 💡 **Warning:** This is diffefent `mmproj` projector! If the projector didn't freeze during fine-tune, it may have degraded (the vector space "floated"). In this case, there is a 95% chance that the projector is not damaged.

For example:
`Cydonia-24B-v4.3-absolute-heresy.IQ4_XS.gguf` + `mmproj-Mistral-Small-3.1-24B-Instruct-2503-f16.gguf`

> 💡 **Warning:** I couldn't find a compatible chat handler, so I'm using a custom one. 

```json
{
    "model_path": "H:\\LLM2\\Cydonia_24b\\Cydonia-24B-v4.3-absolute-heresy.IQ4_XS.gguf",
    "mmproj_path": "H:\\LLM2\\Cydonia_24b\\mmproj-Mistral-Small-3.1-24B-Instruct-2503-f16.gguf",
    "top_p": 0.9,
    "min_p": 0.02,
    "top_k": 40,
    "chat_handler": "llava15",
    "raw_mode": true,
    "prompt_template": "[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{images}{user}[/INST]",
    "stop": "[\"</s>\",\"[INST]\",\"[SYSTEM_PROMPT]\"]"
}
```

</details>


<details>

<summary>Qwen3.5-9B</summary>

- https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF
- https://huggingface.co/unsloth/Qwen3.5-2B-GGUF
- https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
- https://huggingface.co/unsloth/Qwen3.5-9B-GGUF

For example:
`Qwen3.5-9B-Q4_K_M.gguf` + `mmproj-BF16.gguf`

And a new option appeared `enable_thinking": true`, - If you want the model to think (this may give a better result), write true, but this will take more time and require more context, plus the `think` section will have to be cut off later.

Other parameters should be selected based on recommendations, based on the task, or empirically, as you prefer.

```json
{
    "model_path": "I:\\LLM\\qwen\\qwen35-9b\\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
    "mmproj_path": "I:\\LLM\\qwen\\qwen35-9b\\mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": 20,
    "repeat_penalty": 1.0,
    "chat_handler": "qwen35",
    "image_min_tokens": 1024,
    "image_max_tokens": 2048
}
```

</details>

<details>

<summary>Qwen3-VL-8B</summary>

- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
- https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-abliterated-v2.0-GGUF

For example:
`Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0.gguf` + `Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-Q8_0.gguf`

```json
{
    "model_path": "H:\\LLM2\\Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0.gguf",
    "mmproj_path": "H:\\LLM2\\Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-Q8_0.gguf",
    "min_p": 0.01,
    "top_k": 40,
    "chat_handler": "qwen3",
    "image_min_tokens": 1024,
    "image_max_tokens": 2048
}
```

</details>

<details>

<summary>Gemma3-12B</summary>

- https://huggingface.co/unsloth/gemma-3-12b-it-GGUF
  
For example: `gemma-3-12b-it-Q4_K_M.gguf` + `mmproj-BF16.gguf`

```json
{
    "model_path": "H:\\LLM2\\gemma3_12b\\gemma-3-12b-it-Q4_K_M.gguf",
    "mmproj_path": "H:\\LLM2\\gemma3_12b\\mmproj-BF16.gguf",
    "n_batch": 4096,
    "top_p": 0.95,
    "min_p": 0.01,
    "repeat_penalty": 1.0,
    "chat_handler": "gemma3",
    "image_min_tokens": 256,
    "image_max_tokens": 256
}
```

</details>

<details>

<summary>Joycaption-Beta</summary>

- https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf/tree/main

For example:
`llama-joycaption-beta-one-hf-llava-q8_0.gguf` + `llama-joycaption-beta-one-llava-mmproj-model-f16.gguf`

> 💡 **Tip:** This model likes it when the task is written in `user_prompt`, so we use the option `"system_preset_to_user_prompt": true`. The system prompt is always the same `"system_prompt_default": "You are a helpful image captioner."` - set this text as the default value. The model requires a special prompt template. So, enable `"raw_mode": true`. This will set the new `prompt_template` and `stop` words for this model. With these parameters, the model will stop sticking, communicating with itself (with the assistant) and will strictly follow the prompt.

```json
{
    "model_path": "H:\\LLM2\\joycaption-beta\\llama-joycaption-beta-one-hf-llava-q8_0.gguf",
    "mmproj_path": "H:\\LLM2\\joycaption-beta\\llama-joycaption-beta-one-llava-mmproj-model-f16.gguf",
    "n_ctx": 2048,
    "n_batch": 1024,
    "max_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.9,
    "min_p": 0.01,
    "top_k": 40,
    "repeat_penalty": 1.2,
    "chat_handler": "llava15",
    "system_prompt_default": "You are a helpful image captioner.",
    "system_preset_to_user_prompt": true,
    "raw_mode": true,
    "prompt_template": "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{images}{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "stop": "[\"<|eot_id|>\",\"<|end_of_text|>\"]"
}
```

</details>

<details>

<summary>Ministral-3-14B</summary>

- https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-GGUF/tree/main

For example:
`Ministral-3-14B-Instruct-2512-Q4_K_M.gguf` + `Ministral-3-14B-Instruct-2512-BF16-mmproj.gguf`

```json
{
    "model_path": "H:\\LLM2\\Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
    "mmproj_path": "H:\\LLM2\\Ministral-3-14B-Instruct-2512-BF16-mmproj.gguf",
    "temperature": 0.3,
    "min_p": 0.01,
    "top_k": 40,
    "chat_handler": "llava15",
    "raw_mode": true,
    "prompt_template": "[INST]{system}\\n\\n{images}{user}[/INST]",
    "stop": "[\"</s>\",\"[INST]\",\"[/INST]\"]",
    "image_min_tokens": 1024,
    "image_max_tokens": 1024
}
```

</details>

<details>

<summary>Mistral-Nemo-Instruct-2407-Q8(text)</summary>

- https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF

For example: `Mistral-Nemo-Instruct-2407-Q8_0.gguf`

```json
{
    "model_path": "H:\\LLM2\\Mistral-Nemo-Instruct-2407-Q8_0.gguf",
    "max_tokens": 1536,
    "temperature": 0.3,
    "min_p": 0.01,
    "top_k": 40,
    "chat_format": "mistral-instruct",
    "force_mmproj": false
}
```

</details>

<details>

<summary>Qwen3-4b-Z-Engineer-V2(text)</summary>

- https://huggingface.co/BennyDaBall/qwen3-4b-Z-Image-Engineer
  
For example: `Qwen3-4b-Z-Engineer-V2.gguf`

```json
{
    "model_path": "I:\\LLM\\qwen\\Qwen3-4b-Z-Engineer-V2.gguf",
    "n_ctx": 4096,
    "min_p": 0.01,
    "chat_format": "qwen",
    "force_mmproj": false
}
```

</details>

<details>

<summary>BGE-M3-Q4_K_M (encoder)</summary>

A fast encoder that allows you to obtain text embeddings that can then be used for searching in vector databases.

- https://huggingface.co/groonga/bge-m3-Q4_K_M-GGUF
  
For example: `bge-m3-q4_k_m.gguf`

```json
{
    "model_path": "I:\\LLM\\encoder\\bge\\bge-m3-q4_k_m.gguf",
    "n_ctx": 2048,
    "force_mmproj": false,
    "extract_embedding": true,
    "pooling_type": 1
}
```

</details>

<details>

<summary>Z-Qwen_3_4b-Q8_0 (encoder)</summary>

- https://huggingface.co/Qwen/Qwen3-4B-GGUF
  
For example: `Qwen_3_4b-Q8_0.gguf`

> 💡 **Warning:** An important limitation. llama.cpp doesn't allow you to retrieve the -2 hidden layer needed for this model. It always outputs the last layer. Therefore, the vectors don't match those generated by comfy-ui or HF.

> 💡 **Warning:** This encoder has a corrupted built-in tokenizer that doesn't handle system tokens correctly. So, I added the ability to override the tokenizer. You can download it here
https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/tokenizer. 

```json
{
    "model_path": "H:\\Stable_Diffusion\\text_encoders\\Z\\Qwen_3_4b-Q8_0.gguf",
    "n_ctx": 2048,
    "prompt_template": "<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n",
    "force_mmproj": false,
    "extract_embedding": true,
    "tokenizer_path": "I:\\LLM\\encoder\\Z-Image-Turbo-HF\\tokenizer",
    "embedding_scale": 100.0,
    "convert_emb_to_cond": true
}
```

</details>

---

# Speed test and memory overflow problem:

<img width="2048" height="448" alt="03458-310245416557914" src="https://github.com/user-attachments/assets/ed94d57c-5050-4fdf-b41c-688cfc88e09e" />

LLM and CLIP cannot be split (as can be done with UNET). They must be loaded in their entirety.
But if the model is MoE, you can unload some of the experts into RAM so that they can be processed by the CPU. This way you can run large models.

In any case, make sure your VRAM doesn't overflow. If you allow your VRAM to overflow, some layers will be loaded into slower RAM, the GPU will be forced to read from RAM, which will inevitably lead to a 5-7x performance degradation!

Open **Task Manager** (Ctrl+Alt+Del) → Performance tab → GPU → set 'CUDA' engine graph. Check the memory usage during execution in middle graph. It shouldn't exceed the VRAM memory limit. Even nearing the upper limit can be considered overflow, which will cause catastrophic performance slowdowns. And in some cases, even to a crash with an **OOM (out of memory)** error.
GPU drivers often reserve a small amount of VRAM for system needs, so 100% VRAM usage will not be possible.

Model fits (good speed) ✅:

<img width="439" height="438" alt="image" src="https://github.com/user-attachments/assets/d463c17c-f591-436b-b524-f9cce2aad993" />

The bottom graph (shared memory) should be empty!

> 💡 **Nuance:** When using `use_mmap=false` the operating system may use RAM for file caching, which Task Manager may display as "used" shared memory, but this does not always mean that VRAM is full.

Memory overflow (speed down ) ❌:

<img width="450" height="434" alt="image" src="https://github.com/user-attachments/assets/f44905f2-b6b5-4e6b-b1eb-c922f643972c" />

VRAM reached its maximum and then shared memory started to fill up → performance degradation.

| Mode | Speed for Qwen3.6-35B-A3B-Q4_K_M in 16 Gb VRAM | Note |
|--------|--------|--------|
| n_cpu_moe | 50-60 tok/sec | llama.cpp build from source with `AVX, AVX2, AVX512` |
| NGL | 29 tok/sec  | llama.cpp build from source with `AVX, AVX2, AVX512` |
| Memory overflow ❌ | 10.8 tok/sec | llama.cpp build from source with `VMM` |

> 💡 **WARNING:** These ready-made **basic** VHLs may not have CPU acceleration (AVX, AVX2, AVX512) implementations. Therefore, installing them may not yield any benefit from `n_cpu_moe` or `cpu_moe`. Use VHL with optimizations enabled, or better yet, compile the project yourself for your hardware. Also, ready-made VHLs may not contain VMM (Virtual Memory Management), which will lead to a crash with an OOM (out of memory) error in case of insufficient VRAM.

| Mode | Speed for Qwen3.6-35B-A3B-Q4_K_M in 16 Gb VRAM | Note |
|--------|--------|--------|
| n_cpu_moe | 20-30 tok/sec | 💡 llama.cpp from ready-made basic VHLs **without** `AVX, AVX2, AVX512` |
| Memory overflow ❌ | **OOM crash** | 💡 llama.cpp from ready-made basic VHLs **without** `VMM` |

> 💡 **Tip:** Search for models on huggingface and choose models with better quantization, such as UD_IQ from unsloth. They will be smarter and lighter.

To make the model fit:
1. Use stronger quantization Q8->Q6->Q4->Q3... (But the stronger the quantization, the more the quality of the model may suffer; below Q4 it may already be unacceptable.)
2. Reduce `n_ctx`, but not too much, otherwise the response may be cut off.
3. In a larger context enable KV cache quantization `"type_k": 8`, `"type_v": 8`
4. Use MoE model with expert unloading (n_cpu_moe > 0 or cpu_moe = true and n_gpu_layers=-1). Some experts will be stored in RAM and processed by the CPU. This is a more efficient method than NGL.
- n_cpu_moe = 20 (You need to choose the best number) → put 20 experts on CPU, rest on GPU → All available VRAM is full, higher speed.
- cpu_moe = true → All experts on CPU → minimal VRAM consumption.
5. If nothing else is possible use NGL offload (n_gpu_layers > 0). Some layers will be stored in RAM and processed by the CPU.
- n_gpu_layers = -1 → try to put ALL layers on GPU (if VRAM allows)
- n_gpu_layers = 22 (You need to choose the best number) → put 22 layers on GPU, rest on CPU. 
- n_gpu_layers = 0 → all layers on CPU (slower)

Please note that in addition to the model weights, you also need to fit the mmproj projector into memory.

Please note that in addition to the model and projector weights, you also need to fit the KV cache into memory. Increasing the context increases the KV cache size.

If the memory is full before this node starts use `unload_all_models = true`.

If `debug=true` this node in calculates in console the generation time (tok/sec) from the start of inference to its completion, which also includes overhead such as graph compilation/optimization, vision encoder preprocessing (if applicable), prompt tokenization & embedding, VRAM allocation, sampling/decoding initialization etc.
LM Studio displays the net generation time, so the values in LM Studio will be higher (better tok/sec).
You can view the net generation time (`eval time` in llama.cpp verbose output) in console by enabling `verbose=true`.

---

## Troubleshooting:

<img width="2048" height="448" alt="03528-1060011778618551" src="https://github.com/user-attachments/assets/ce5e50f4-131f-4f4e-959e-f9890d32b2fc" />

Try enabling debug output:
```
"debug": true
```
Watch the ComfyUI console for HTTP errors from llama-server.

<details>

<summary>troubleshooting</summary>

### 1. Issue: llama-server is not reachable

The node cannot connect to `server_url` (default `http://127.0.0.1:8080`).

- Start `llama-server` / `llama-server.exe` first
- Check `--port` matches `server_url`
- `curl http://127.0.0.1:8080/health` should return `{"status":"ok"}`
- HTTP 503 = the GGUF is still loading; wait and retry

### 2. Issue: HTTP 500 / empty / garbage vision answers

- Confirm llama-server was started with **both** `-m` and `--mmproj`
- Use a **current** official llama.cpp build (Qwen3-VL needs a recent mmproj)
- Increase `--ctx-size` (`-c`) if the prompt + image tokens overflow
- For thinking models, start with `--jinja` and toggle `enable_thinking` on the node

### 3. Issue: GPU / VRAM

GPU offload is a llama-server flag, not a ComfyUI widget:

```bash
llama-server -m model.gguf --mmproj mmproj.gguf -ngl 99 -c 8192 --port 8080
```

Lower `-ngl` or `-c` if the server OOMs. Stopping llama-server frees that VRAM for diffusion.

### 4. Issue: ggml_new_object / context too small

This now happens **inside llama-server**. Increase `-c` / `--ctx-size` or lower image resolution / `max_frames`. The old `pool_size` Python setting is not sent over HTTP.

</details>

---

Maybe it will be useful to someone.

[!] Tested against the OpenAI-compatible llama-server API from current `ggml-org/llama.cpp`. Hardware notes in older versions referred to Windows + RTX 5080/2060.

# Dependencies & Thanks:
- https://github.com/ggml-org/llama.cpp (`llama-server`, `libmtmd`, Qwen3-VL)
- https://huggingface.co/Qwen

No `llama-cpp-python`.
